"""
Migrate legacy FalkorDB graph → graph_zero schema.

Mapping:
  Legacy TerrainNode → graph_zero TerrainNode
    terrain_role "bedrock"  → layer "bedrock"
    terrain_role "living"   → layer "community"
    courage → compassion (virtue index remap)
    source_work, author → preserved as properties
    embedding (1536-dim) → preserved
    virtue_* scores → stored as properties
    momentum_* → stored as properties
    kala_accumulated, traversal_count → preserved

  Legacy Virtue (20) → graph_zero VirtueAnchor (9)
    Already bootstrapped, skip

  Legacy VesselAnchor (40) → inspect and migrate relevant ones
  Legacy Agent (5) → create as vessels in graph_zero

Runs as a POST endpoint so it executes inside Railway.
"""

import os
import time
from falkordb import FalkorDB

# Graph Zero virtue order (canonical)
GZ_VIRTUES = ["unity", "justice", "truthfulness", "love", "detachment",
              "humility", "compassion", "wisdom", "service"]

# Legacy virtue properties
LEGACY_VIRTUES = ["unity", "justice", "truthfulness", "love", "detachment",
                  "humility", "service", "courage", "wisdom"]

# Map legacy → GZ: courage → compassion
VIRTUE_MAP = {
    "unity": "unity", "justice": "justice", "truthfulness": "truthfulness",
    "love": "love", "detachment": "detachment", "humility": "humility",
    "service": "service", "courage": "compassion", "wisdom": "wisdom"
}

ROLE_TO_LAYER = {
    "bedrock": "bedrock",
    "living": "community",
    None: "community",
}


def escape(s):
    if s is None: return "null"
    return s.replace("\\", "\\\\").replace("'", "\\'")


def migrate_terrain(legacy_graph, gz_graph, batch_size=100):
    """Migrate TerrainNodes from legacy to graph_zero."""
    stats = {"migrated": 0, "skipped": 0, "errors": 0, "batches": 0}

    # Count total
    total_r = legacy_graph.query("MATCH (n:TerrainNode) RETURN count(n)")
    total = total_r.result_set[0][0]
    stats["total_legacy"] = total

    offset = 0
    while offset < total:
        try:
            # Fetch batch (SKIP/LIMIT with ORDER BY internal ID)
            result = legacy_graph.query(
                f"MATCH (n:TerrainNode) RETURN n ORDER BY ID(n) SKIP {offset} LIMIT {batch_size}")

            if not result.result_set:
                break

            for row in result.result_set:
                node = row[0]
                props = dict(node.properties)
                node_id = props.get("id", f"legacy_{offset}")

                # Check if already migrated
                check = gz_graph.query(f"MATCH (n:TerrainNode {{_id: '{escape(node_id)}'}}) RETURN count(n)")
                if check.result_set[0][0] > 0:
                    stats["skipped"] += 1
                    continue

                # Map properties
                text = props.get("text", "")
                source_work = props.get("source_work", "")
                author = props.get("author", "")
                terrain_role = props.get("terrain_role", "living")
                layer = ROLE_TO_LAYER.get(terrain_role, "community")

                # Build virtue position (in GZ order)
                virtue_pos = []
                for gv in GZ_VIRTUES:
                    # Find legacy key that maps to this GZ virtue
                    legacy_key = None
                    for lk, gk in VIRTUE_MAP.items():
                        if gk == gv:
                            legacy_key = lk
                            break
                    val = props.get(f"virtue_{legacy_key}", 0.5) if legacy_key else 0.5
                    virtue_pos.append(float(val))

                # Momentum
                momentum = []
                for gv in GZ_VIRTUES:
                    legacy_key = None
                    for lk, gk in VIRTUE_MAP.items():
                        if gk == gv:
                            legacy_key = lk
                            break
                    val = props.get(f"momentum_{legacy_key}", 0.0) if legacy_key else 0.0
                    momentum.append(float(val))

                kala = float(props.get("kala_accumulated", 0.0))
                traversals = int(props.get("traversal_count", 0))

                # Embedding (1536 floats) — store as list property
                embedding = props.get("embedding", [])

                # Build Cypher MERGE
                # Can't pass list params easily, so we'll set embedding separately
                prop_str = (
                    f"_id: '{escape(node_id)}', "
                    f"text: '{escape(text)}', "
                    f"source_work: '{escape(source_work)}', "
                    f"author: '{escape(author)}', "
                    f"layer: '{escape(layer)}', "
                    f"terrain_role: '{escape(terrain_role)}', "
                    f"kala_accumulated: {kala}, "
                    f"traversal_count: {traversals}, "
                    f"provenance_type: 'BEDROCK'"
                )

                # Add virtue scores as individual properties
                for i, gv in enumerate(GZ_VIRTUES):
                    prop_str += f", virtue_{gv}: {virtue_pos[i]}"
                for i, gv in enumerate(GZ_VIRTUES):
                    prop_str += f", momentum_{gv}: {momentum[i]}"

                try:
                    gz_graph.query(
                        f"MERGE (n:TerrainNode {{_id: '{escape(node_id)}'}}) "
                        f"SET n = {{{prop_str}}}")

                    # Set embedding as a separate query (list handling)
                    if embedding and isinstance(embedding, list) and len(embedding) > 0:
                        # FalkorDB supports list properties
                        emb_str = "[" + ",".join(str(float(x)) for x in embedding[:1536]) + "]"
                        gz_graph.query(
                            f"MATCH (n:TerrainNode {{_id: '{escape(node_id)}'}}) "
                            f"SET n.embedding = {emb_str}")

                    stats["migrated"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        stats[f"error_sample_{stats['errors']}"] = f"{node_id}: {str(e)[:100]}"

            stats["batches"] += 1
            offset += batch_size

            # Progress
            if stats["batches"] % 10 == 0:
                print(f"  Migration progress: {offset}/{total} "
                      f"({stats['migrated']} migrated, {stats['skipped']} skipped, {stats['errors']} errors)")

        except Exception as e:
            stats["batch_error"] = f"Batch at offset {offset}: {str(e)[:200]}"
            stats["errors"] += 1
            offset += batch_size  # Skip failed batch

    return stats


def migrate_agents(legacy_graph, gz_graph):
    """Migrate legacy Agent nodes to graph_zero vessels."""
    stats = {"migrated": 0, "skipped": 0}

    result = legacy_graph.query("MATCH (a:Agent) RETURN a")
    for row in result.result_set:
        agent = row[0]
        props = dict(agent.properties)
        agent_id = props.get("id", "unknown")
        name = props.get("name", agent_id)

        # Check if exists
        check = gz_graph.query(f"MATCH (n:VesselAnchor {{_id: '{escape(agent_id)}'}}) RETURN count(n)")
        if check.result_set[0][0] > 0:
            stats["skipped"] += 1
            continue

        # Create vessel anchor + position + momentum + kala
        gz_graph.query(
            f"CREATE (n:VesselAnchor {{_id: '{escape(agent_id)}', name: '{escape(name)}', "
            f"agent_type: 'ai', attestation_depth: 1, created: {int(time.time())}}})")

        gz_graph.query(
            f"CREATE (n:VesselPosition {{_id: 'pos_{escape(agent_id)}', "
            f"position: [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}})")

        gz_graph.query(
            f"CREATE (n:VesselMomentum {{_id: 'mom_{escape(agent_id)}', "
            f"momentum: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}})")

        gz_graph.query(
            f"CREATE (n:VesselKala {{_id: 'kala_{escape(agent_id)}', balance: 100.0}})")

        # Link them
        gz_graph.query(
            f"MATCH (a:VesselAnchor {{_id: '{escape(agent_id)}'}}), "
            f"(p:VesselPosition {{_id: 'pos_{escape(agent_id)}'}}) "
            f"CREATE (p)-[:PART_OF_VESSEL {{_id: 'e_pos_{escape(agent_id)}', "
            f"_source_id: 'pos_{escape(agent_id)}', _target_id: '{escape(agent_id)}'}}]->(a)")

        gz_graph.query(
            f"MATCH (a:VesselAnchor {{_id: '{escape(agent_id)}'}}), "
            f"(m:VesselMomentum {{_id: 'mom_{escape(agent_id)}'}}) "
            f"CREATE (m)-[:PART_OF_VESSEL {{_id: 'e_mom_{escape(agent_id)}', "
            f"_source_id: 'mom_{escape(agent_id)}', _target_id: '{escape(agent_id)}'}}]->(a)")

        gz_graph.query(
            f"MATCH (a:VesselAnchor {{_id: '{escape(agent_id)}'}}), "
            f"(k:VesselKala {{_id: 'kala_{escape(agent_id)}'}}) "
            f"CREATE (k)-[:PART_OF_VESSEL {{_id: 'e_kala_{escape(agent_id)}', "
            f"_source_id: 'kala_{escape(agent_id)}', _target_id: '{escape(agent_id)}'}}]->(a)")

        # Link to community
        gz_graph.query(
            f"MATCH (a:VesselAnchor {{_id: '{escape(agent_id)}'}}), "
            f"(c:Community) "
            f"CREATE (a)-[:MEMBER_OF {{_id: 'e_mem_{escape(agent_id)}', "
            f"_source_id: '{escape(agent_id)}', _target_id: c._id}}]->(c)")

        stats["migrated"] += 1

    return stats
