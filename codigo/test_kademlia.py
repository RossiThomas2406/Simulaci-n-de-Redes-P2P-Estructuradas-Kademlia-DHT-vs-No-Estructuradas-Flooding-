import pytest
import random
from simulador_p2p_sensibilidad_k import KademliaNode, KademliaNetwork


def test_construccion_buckets_lrs():
    """Verifica que la política Least-Recently Seen (LRS) expulse e integre nodos correctamente."""
    node = KademliaNode(node_id=0, k=3)
    active_nodes = {0, 8, 9, 10, 11, 12}

    node.add_contact(8, active_nodes)
    node.add_contact(9, active_nodes)
    node.add_contact(10, active_nodes)

    assert len(node.buckets[3]) == 3
    assert node.buckets[3] == [8, 9, 10]

    # Comprobación de vida
    node.add_contact(11, active_nodes)
    assert node.buckets[3] == [9, 10, 8]

    # Reemplazo por desconexión
    active_nodes.remove(9)
    node.add_contact(12, active_nodes)
    assert node.buckets[3] == [10, 8, 12]


def test_hit_local_inmediato():
    """Verifica que si el iniciador ya tiene el recurso, resuelve con 0 mensajes y 0 saltos."""
    net = KademliaNetwork(N=20, k=3, alpha=3)
    initiator = list(net.active_node_ids)[0]
    target_key = 999

    # Inyectamos el recurso directamente en el storage del iniciador
    net.nodes[initiator].storage[target_key] = "LocalHitData"

    found, msgs, hops, closest = net.iterative_lookup(
        initiator, target_key, is_publishing=False
    )
    assert found is True
    assert msgs == 0
    assert hops == 0
    assert closest == [initiator]


def test_replicacion_tolerancia_parcial_churn():
    """Verifica que la red resista la caída parcial de réplicas pero falle si caen todas las k réplicas."""
    net = KademliaNetwork(N=30, k=3, alpha=3)
    initiator = list(net.active_node_ids)[0]
    target_key = 12345

    net.publish_key(initiator, target_key, "TestChurnPayload")

    # Identificamos los nodos que recibieron la réplica
    nodos_con_replica = [
        nid for nid, node in net.nodes.items() if target_key in node.storage
    ]
    assert len(nodos_con_replica) == 3

    # Simulamos caída de k-1 réplicas (quedará 1 activa)
    net.active_node_ids.remove(nodos_con_replica[0])
    net.active_node_ids.remove(nodos_con_replica[1])

    searcher = list(net.active_node_ids)[1]
    found_parcial, _, _, _ = net.iterative_lookup(searcher, target_key)
    assert found_parcial is True  # Sobrevive

    # Simulamos caída total (eliminamos la última réplica)
    net.active_node_ids.remove(nodos_con_replica[2])
    found_total, _, _, _ = net.iterative_lookup(searcher, target_key)
    assert found_total is False  # Colapsa


def test_convergencia_churn():
    """Verifica la estabilidad topológica de la red tras aplicar desconexiones masivas."""
    net = KademliaNetwork(N=100, k=8, alpha=3)
    ids_originales = set(net.active_node_ids)

    net.apply_churn(20)
    ids_nuevos = set(net.active_node_ids)

    assert len(ids_nuevos) == 100
    assert len(ids_originales - ids_nuevos) == 20


def test_convergencia_lookup_top_k():
    """Verifica que iterative_lookup converja sobre los k nodos más cercanos sin recorrer toda la red."""
    random.seed(42)
    net = KademliaNetwork(N=200, k=8, alpha=3)
    initiator = list(net.active_node_ids)[0]
    target_key = random.getrandbits(160)

    found, msgs, hops, closest = net.iterative_lookup(
        initiator, target_key, is_publishing=True
    )

    assert len(closest) == 8
    # Debe converger consultando una fracción acotada de la red, no los 200 nodos
    assert msgs < 100
    # Los k nodos devueltos deben estar ordenados por distancia XOR ascendente
    distancias = [nid ^ target_key for nid in closest]
    assert distancias == sorted(distancias)
    # Al comparar contra el top-k real de toda la red, el más cercano debe coincidir
    top_k_global = sorted(net.active_node_ids, key=lambda nid: nid ^ target_key)[:8]
    assert closest[0] in top_k_global
