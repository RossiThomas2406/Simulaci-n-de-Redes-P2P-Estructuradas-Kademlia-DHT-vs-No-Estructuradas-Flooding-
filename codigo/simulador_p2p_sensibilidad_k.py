"""
TRABAJO FINAL INTEGRADOR - SISTEMAS OPERATIVOS Y REDES 2 (SOR2)
Simulador por Eventos Discretos de Kademlia (DHT)
Estudio de Sensibilidad del Parámetro k (Overhead vs. Resiliencia)
Autor: Thomas Hernán Rossi
"""

import random
import math
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Semillas para reproducibilidad científica
random.seed(42)
np.random.seed(42)

# Configuración estética de Matplotlib para gráficos de calidad académica
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class KademliaNode:
    """Representa un nodo individual dentro de la red Kademlia P2P."""

    def __init__(self, node_id, k, B=160):
        self.node_id = node_id
        self.k = k
        self.B = B
        self.buckets = [[] for _ in range(B)]
        self.storage = {}
        self.maintenance_msgs = 0  # Conteo de mensajes de mantenimiento (PINGs)

    def distance(self, other_id):
        """Métrica de distancia XOR fundamental de Kademlia."""
        return self.node_id ^ other_id

    def _bucket_index(self, other_id):
        dist = self.distance(other_id)
        if dist == 0:
            return 0
        return dist.bit_length() - 1

    def add_contact(self, contact_id, active_nodes_set):
        """Agrega o actualiza un contacto aplicando la política de reemplazo LRS."""
        if contact_id == self.node_id:
            return

        idx = self._bucket_index(contact_id)
        bucket = self.buckets[idx]

        # Si ya existe, lo movemos al final (más reciente)
        if contact_id in bucket:
            bucket.remove(contact_id)
            bucket.append(contact_id)
        # Si hay espacio, simplemente lo agregamos
        elif len(bucket) < self.k:
            bucket.append(contact_id)
        # Si está lleno, aplicamos política LRS
        else:
            oldest_contact = bucket[0]
            self.maintenance_msgs += (
                1  # Simulamos el costo del mensaje PING de mantenimiento
            )

            if oldest_contact in active_nodes_set:
                # Comprobación de vida: Responde, se queda y va al final. El nuevo se descarta.
                bucket.remove(oldest_contact)
                bucket.append(oldest_contact)
            else:
                # Reemplazo por desconexión: No responde (timeout), se elimina y entra el nuevo.
                bucket.pop(0)
                bucket.append(contact_id)

    def find_closest_contacts(self, target_id, count=None):
        """Retorna los 'count' contactos más cercanos al target_id según la distancia XOR."""
        if count is None:
            count = self.k
        all_contacts = []
        for b in self.buckets:
            all_contacts.extend(b)
        all_contacts.sort(key=lambda cid: cid ^ target_id)
        return all_contacts[:count]


class KademliaNetwork:
    """Simula el grafo de la red Kademlia, el ruteo iterativo y la replicación."""

    def __init__(self, N, k, alpha=3, B=160):
        self.N = N
        self.k = k
        self.alpha = alpha
        self.B = B
        self.nodes = {}
        self.active_node_ids = set()

        # Generar IDs únicos
        while len(self.active_node_ids) < N:
            nid = random.getrandbits(B)
            if nid not in self.nodes:
                self.nodes[nid] = KademliaNode(nid, k, B)
                self.active_node_ids.add(nid)

        # Poblar las tablas de ruteo iniciales (Bootstrap)
        all_ids = list(self.active_node_ids)
        for nid, node in self.nodes.items():
            sample_size = min(len(all_ids), k * 10)
            sample = random.sample(all_ids, sample_size)
            for other_id in sample:
                node.add_contact(other_id, self.active_node_ids)

    def reset_maintenance_counters(self):
        """Limpia los contadores de inicialización para medir solo el costo operativo."""
        for node in self.nodes.values():
            node.maintenance_msgs = 0

    def apply_churn(self, churn_pct):
        """Simula churn balanceado: Desconexión de un lote e incorporación inmediata."""
        if churn_pct <= 0:
            return

        num_to_churn = int(self.N * (churn_pct / 100.0))

        # 1. Salidas
        removed_ids = set(random.sample(list(self.active_node_ids), num_to_churn))
        self.active_node_ids -= removed_ids

        # 2. Entradas
        for _ in range(num_to_churn):
            new_id = random.getrandbits(self.B)
            while new_id in self.nodes:
                new_id = random.getrandbits(self.B)

            new_node = KademliaNode(new_id, self.k, self.B)
            self.nodes[new_id] = new_node
            self.active_node_ids.add(new_id)

            # Bootstrap del nuevo nodo
            bootstrap_sample = random.sample(
                list(self.active_node_ids), min(self.k * 2, len(self.active_node_ids))
            )
            for other_id in bootstrap_sample:
                new_node.add_contact(other_id, self.active_node_ids)

    def publish_key(self, initiator_id, key_id, value="DataPayload"):
        """Búsqueda de los k más cercanos y envío del STORE (réplicas)."""
        _, lookup_msgs, _, closest_nodes = self.iterative_lookup(
            initiator_id, key_id, is_publishing=True
        )

        store_msgs = 0
        for r_id in closest_nodes:
            if r_id in self.active_node_ids:
                self.nodes[r_id].storage[key_id] = value
                store_msgs += 1

        return lookup_msgs, store_msgs

    def iterative_lookup(self, initiator_id, target_key, is_publishing=False):
        """Búsqueda iterativa alpha-concurrente con resolución local y convergencia explícita en top-k."""
        if initiator_id not in self.active_node_ids:
            return False, 0, 0, []

        initiator_node = self.nodes[initiator_id]

        # Resolución local inmediata si el iniciador ya posee la clave y no está publicando
        if not is_publishing and target_key in initiator_node.storage:
            return True, 0, 0, [initiator_id]

        # Candidatos iniciales: los k contactos más cercanos conocidos por el iniciador
        initial_contacts = initiator_node.find_closest_contacts(target_key, self.k)
        candidates_set = set(initial_contacts)
        candidates_set.add(initiator_id)  # El iniciador también es candidato a réplica

        visited = set([initiator_id])
        dead = set()
        lookup_msgs = 0
        hops = 0
        found = False

        while True:
            # Ordenamos los candidatos no descartados por distancia XOR al objetivo
            live_candidates = sorted(
                (c for c in candidates_set if c not in dead),
                key=lambda cid: cid ^ target_key,
            )

            # Criterio explícito de convergencia: verificamos los k nodos más cercanos actuales
            top_k = live_candidates[: self.k]
            unvisited_top_k = [c for c in top_k if c not in visited]

            # Si todos los k nodos más cercanos ya fueron consultados, la búsqueda convergió
            if not unvisited_top_k:
                break

            hops += 1
            query_batch = unvisited_top_k[: self.alpha]

            for queried_id in query_batch:
                visited.add(queried_id)
                lookup_msgs += 1

                if queried_id not in self.active_node_ids:
                    # Nodo inactivo: se marca como muerto y se purga reactivamente del bucket
                    dead.add(queried_id)
                    idx = initiator_node._bucket_index(queried_id)
                    if queried_id in initiator_node.buckets[idx]:
                        initiator_node.buckets[idx].remove(queried_id)
                    continue

                initiator_node.add_contact(queried_id, self.active_node_ids)
                queried_node = self.nodes[queried_id]

                if not is_publishing and target_key in queried_node.storage:
                    found = True
                    break

                contacts_returned = queried_node.find_closest_contacts(
                    target_key, self.k
                )
                for c in contacts_returned:
                    if c not in dead:
                        candidates_set.add(c)

            if found:
                break

        final_closest = sorted(
            (c for c in candidates_set if c in self.active_node_ids),
            key=lambda cid: cid ^ target_key,
        )
        return found, lookup_msgs, hops, final_closest[: self.k]


def run_experiment_suite():
    print("=" * 75)
    print(" INICIANDO SIMULACIÓN P2P KADEMLIA: ESTUDIO DE SENSIBILIDAD k")
    print("=" * 75)

    N_values = [1000, 5000]
    k_values = [3, 8, 14, 20]
    churn_levels = [0, 10, 20, 30]
    alpha_fixed = 3
    queries_per_run = 150
    reps = 10

    results_summary = []
    raw_rows = []
    raw_data = {}

    for N in N_values:
        for k in k_values:
            for churn in churn_levels:
                print(
                    f"Ejecutando: N={N}, k={k}, churn={churn}% ({reps} repeticiones)...",
                    end=" ",
                    flush=True,
                )

                reps_recall = []
                reps_overhead = []
                reps_hops = []

                for rep_idx in range(reps):
                    net = KademliaNetwork(N=N, k=k, alpha=alpha_fixed)

                    # 1. Separación: Limpiamos los mensajes del bootstrap/inicialización
                    net.reset_maintenance_counters()
                    active_nodes = list(net.active_node_ids)

                    # 2. Fase de Publicación
                    keys_to_query = []
                    tot_pub_lookups, tot_pub_stores = 0, 0
                    for _ in range(queries_per_run):
                        key_id = random.getrandbits(160)
                        initiator = random.choice(active_nodes)
                        p_look, p_store = net.publish_key(
                            initiator, key_id, value="Data"
                        )
                        tot_pub_lookups += p_look
                        tot_pub_stores += p_store
                        keys_to_query.append(key_id)

                    # 3. Aplicar Churn
                    net.apply_churn(churn)

                    # 4. Fase de Búsqueda
                    successes = 0
                    total_query_msgs = 0
                    total_hops = 0
                    current_active = list(net.active_node_ids)

                    for key_id in keys_to_query:
                        initiator = random.choice(current_active)
                        found, msgs, hops, _ = net.iterative_lookup(initiator, key_id)
                        total_query_msgs += msgs
                        total_hops += hops
                        if found:
                            successes += 1

                    # 5. Cálculo del overhead limpio
                    tot_maint_msgs = sum(
                        node.maintenance_msgs for node in net.nodes.values()
                    )
                    overhead_total_operativo = (
                        total_query_msgs
                        + tot_pub_lookups
                        + tot_pub_stores
                        + tot_maint_msgs
                    )
                    overhead_por_busqueda = overhead_total_operativo / queries_per_run
                    recall_pct = (successes / queries_per_run) * 100.0
                    avg_hops = total_hops / queries_per_run if successes > 0 else 0

                    reps_recall.append(recall_pct)
                    reps_overhead.append(overhead_por_busqueda)
                    reps_hops.append(avg_hops)

                    # Registro fila a fila para el CSV raw
                    raw_rows.append(
                        {
                            "N": N,
                            "k": k,
                            "churn_pct": churn,
                            "rep_id": rep_idx + 1,
                            "recall_pct": recall_pct,
                            "overhead_per_query": overhead_por_busqueda,
                            "avg_hops": avg_hops,
                            "query_msgs_total": total_query_msgs,
                            "pub_lookup_msgs": tot_pub_lookups,
                            "pub_store_msgs": tot_pub_stores,
                            "maintenance_msgs": tot_maint_msgs,
                        }
                    )

                config_key = f"{N}_{k}_{churn}"
                raw_data[config_key] = {
                    "overhead": reps_overhead,
                    "recall": reps_recall,
                }

                mean_recall = np.mean(reps_recall)
                mean_overhead = np.mean(reps_overhead)
                tradeoff_index = mean_recall / mean_overhead if mean_overhead > 0 else 0

                results_summary.append(
                    {
                        "N": N,
                        "k": k,
                        "churn_pct": churn,
                        "recall_mean": mean_recall,
                        "recall_std": np.std(reps_recall),
                        "overhead_mean": mean_overhead,
                        "overhead_std": np.std(reps_overhead),
                        "hops_mean": np.mean(reps_hops),
                        "hops_std": np.std(reps_hops),
                        "tradeoff_index": tradeoff_index,
                    }
                )
                print("OK")

    # Exportación doble exigida
    pd.DataFrame(raw_rows).to_csv("resultados_sensibilidad_k_RAW.csv", index=False)
    pd.DataFrame(results_summary).to_csv(
        "resultados_sensibilidad_k_RESUMEN.csv", index=False
    )
    # Análisis estadístico (Mann-Whitney U) entre valores consecutivos de k
    raw_df = pd.DataFrame(raw_rows)
    stat_rows = []
    pares_k = [(3, 8), (8, 14), (14, 20)]

    for N in N_values:
        for churn in churn_levels:
            sub = raw_df[(raw_df["N"] == N) & (raw_df["churn_pct"] == churn)]
            for k1, k2 in pares_k:
                g1 = sub[sub["k"] == k1]
                g2 = sub[sub["k"] == k2]
                fila = {"N": N, "churn_pct": churn, "comparacion": f"k={k1} vs k={k2}"}
                for col, pref in [
                    ("recall_pct", "recall"),
                    ("overhead_per_query", "overhead"),
                    ("avg_hops", "hops"),
                ]:
                    v1, v2 = g1[col].values, g2[col].values
                    if np.all(v1 == v2):
                        u_stat, p_val = np.nan, 1.0
                    else:
                        u_stat, p_val = mannwhitneyu(v1, v2, alternative="two-sided")
                    fila[f"{pref}_U"] = (
                        round(float(u_stat), 2) if not np.isnan(u_stat) else None
                    )
                    fila[f"{pref}_p"] = round(float(p_val), 4)
                    fila[f"{pref}_sig"] = "Sí" if p_val < 0.05 else "No"
                stat_rows.append(fila)

    pd.DataFrame(stat_rows).to_csv(
        "resultados_estadisticos_mannwhitney.csv", index=False
    )
    print(
        "[+] Análisis estadístico guardado en 'resultados_estadisticos_mannwhitney.csv'."
    )
    print("\n[+] Datasets 'RAW' (320 iteraciones) y 'RESUMEN' guardados exitosamente.")


if __name__ == "__main__":
    run_experiment_suite()
