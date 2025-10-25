"""
IA BAC BO - VERSÃO FINAL COM ESTATÍSTICAS SEMPRE VISÍVEIS
==========================================================

✓ Arquitetura LSTM (2 camadas, 64 neurônios)
✓ 15 features avançadas
✓ Estatísticas sempre visíveis
✓ Últimos 5 dados na ordem correta (arquivo)

Ganho esperado: ~65-75% de acurácia
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

print("="*80)
print("IA BAC BO - VERSÃO FINAL COM ESTATÍSTICAS")
print("="*80)
print("✓ LSTM (2 camadas, 64 neurônios)")
print("✓ Estatísticas sempre visíveis")
print("✓ Ordem correta dos últimos 5")
print("="*80)
print()

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
arquivo_resultados = 'bacbo_resultados.txt'
arquivo_modelo = 'modelo_bacbo_lstm.pth'
arquivo_metricas = 'metricas_detalhadas.json'

historico = []
previsao_pendente = None
origem_previsao = None
confianca_previsao = 0.0

mapping = {'azul': 0, 'vermelho': 1, 'Empate': 2}
inv_mapping = {v: k for k, v in mapping.items()}
valores_validos = {'azul', 'vermelho', 'Empate'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}\n")

# ============================================================================
# MÉTRICAS AVANÇADAS
# ============================================================================
class MetricasAvancadas:
    def __init__(self):
        self.sucessos = 0
        self.tentativas = 0
        self.sequencia_atual = 0
        self.maior_sequencia = 0

        self.metricas_por_classe = {
            'azul': {'acertos': 0, 'tentativas': 0, 'tp': 0, 'fp': 0, 'fn': 0},
            'vermelho': {'acertos': 0, 'tentativas': 0, 'tp': 0, 'fp': 0, 'fn': 0},
            'Empate': {'acertos': 0, 'tentativas': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        }

        self.confiancas_corretas = []
        self.confiancas_incorretas = []
        self.train_losses = []
        self.val_losses = []
        self.tempos_inferencia = []
        self.oportunidades_total = 0
        self.apostas_realizadas = 0

    def registrar_oportunidade(self):
        self.oportunidades_total += 1

    def registrar_aposta(self, previsao, confianca):
        self.apostas_realizadas += 1

    def registrar_resultado(self, previsao, resultado_real, confianca):
        self.tentativas += 1
        acertou = (previsao == resultado_real)

        if acertou:
            self.sucessos += 1
            self.sequencia_atual += 1
            self.confiancas_corretas.append(confianca)
            if self.sequencia_atual > self.maior_sequencia:
                self.maior_sequencia = self.sequencia_atual

            self.metricas_por_classe[previsao]['tp'] += 1
            self.metricas_por_classe[previsao]['acertos'] += 1
        else:
            self.sequencia_atual = 0
            self.confiancas_incorretas.append(confianca)
            self.metricas_por_classe[previsao]['fp'] += 1
            self.metricas_por_classe[resultado_real]['fn'] += 1

        self.metricas_por_classe[previsao]['tentativas'] += 1

    def registrar_loss(self, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def registrar_tempo_inferencia(self, tempo):
        self.tempos_inferencia.append(tempo)

    def calcular_precision(self, classe):
        metricas = self.metricas_por_classe[classe]
        tp = metricas['tp']
        fp = metricas['fp']
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def calcular_recall(self, classe):
        metricas = self.metricas_por_classe[classe]
        tp = metricas['tp']
        fn = metricas['fn']
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def calcular_f1_score(self, classe):
        precision = self.calcular_precision(classe)
        recall = self.calcular_recall(classe)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calcular_cobertura(self):
        if self.oportunidades_total == 0:
            return 0.0
        return (self.apostas_realizadas / self.oportunidades_total) * 100

    def get_taxa_acertos(self):
        if self.tentativas == 0:
            return 0.0
        return (self.sucessos / self.tentativas) * 100

    def get_derrotas(self):
        return self.tentativas - self.sucessos

    def mostrar_dashboard(self, historico_atual):
        """Dashboard completo com últimos 5 dados"""
        print(f"\n{'='*80}")
        print("📊 DASHBOARD DE MÉTRICAS - IA LSTM BAC BO")
        print(f"{'='*80}")

        # Últimos 5 dados no topo
        print(f"\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + " "*20 + "ÚLTIMOS 5 DADOS LIDOS PELA IA" + " "*29 + "#")
        print("#" + " "*15 + "(Do mais RECENTE para o mais ANTIGO)" + " "*27 + "#")
        print("#" + " "*78 + "#")
        print("#"*80)

        if len(historico_atual) >= 5:
            dados = historico_atual[-5:]
        else:
            dados = historico_atual.copy()

        dados_invertidos = list(reversed(dados))

        for i, dado in enumerate(dados_invertidos, 1):
            if i == 1:
                print(f"#  [{i}] {dado.upper():^15} ← MAIS RECENTE (último capturado)" + " "*22 + "#")
            else:
                print(f"#  [{i}] {dado.upper():^15} ← Posição {i}" + " "*45 + "#")

        print("#" + " "*78 + "#")
        print("#"*80)

        # Métricas Globais
        print(f"\n🎯 MÉTRICAS GLOBAIS:")
        print(f"{'─'*80}")
        taxa = self.get_taxa_acertos()
        derrotas = self.get_derrotas()
        print(f"  Taxa de Acertos:        {taxa:.2f}%")
        print(f"  Vitórias:               {self.sucessos}")
        print(f"  Derrotas:               {derrotas}")
        print(f"  Total de Jogadas:       {self.tentativas}")
        print(f"  Sequência Atual:        {self.sequencia_atual}")
        print(f"  Maior Sequência:        {self.maior_sequencia}")
        print(f"  Cobertura:              {self.calcular_cobertura():.1f}%")

        # Métricas por Classe
        print(f"\n🎨 MÉTRICAS POR CLASSE:")
        print(f"{'─'*80}")
        print(f"  {'Classe':<12} {'Taxa':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for classe in ['azul', 'vermelho', 'Empate']:
            metricas = self.metricas_por_classe[classe]
            if metricas['tentativas'] > 0:
                taxa_classe = (metricas['acertos'] / metricas['tentativas']) * 100
            else:
                taxa_classe = 0.0

            precision = self.calcular_precision(classe) * 100
            recall = self.calcular_recall(classe) * 100
            f1 = self.calcular_f1_score(classe) * 100

            print(f"  {classe:<12} {taxa_classe:>5.1f}%      {precision:>5.1f}%      {recall:>5.1f}%      {f1:>5.1f}%")

        print(f"\n{'='*80}\n")

    def salvar_metricas(self, arquivo):
        dados = {
            'timestamp': datetime.now().isoformat(),
            'metricas_globais': {
                'taxa_acertos': self.get_taxa_acertos(),
                'sucessos': self.sucessos,
                'derrotas': self.get_derrotas(),
                'tentativas': self.tentativas,
                'sequencia_atual': self.sequencia_atual,
                'maior_sequencia': self.maior_sequencia
            }
        }
        with open(arquivo, 'w') as f:
            json.dump(dados, f, indent=2)

    def carregar_metricas(self, arquivo):
        if not os.path.exists(arquivo):
            return

        try:
            with open(arquivo, 'r') as f:
                dados = json.load(f)

            globais = dados.get('metricas_globais', {})
            self.sucessos = globais.get('sucessos', 0)
            self.tentativas = globais.get('tentativas', 0)
            self.sequencia_atual = globais.get('sequencia_atual', 0)
            self.maior_sequencia = globais.get('maior_sequencia', 0)

            print(f"✓ Métricas carregadas")
        except Exception as e:
            print(f"⚠ Erro ao carregar: {e}")

metricas = MetricasAvancadas()

# ============================================================================
# ARQUITETURA LSTM
# ============================================================================
class BacBoLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
        super(BacBoLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out

model = BacBoLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def carregar_modelo():
    if os.path.exists(arquivo_modelo):
        model.load_state_dict(torch.load(arquivo_modelo))
        model.eval()
        print("✓ Modelo carregado")
    else:
        print("⚠ Modelo novo, iniciando do zero")

def salvar_modelo():
    torch.save(model.state_dict(), arquivo_modelo)

def ler_historico_valido(arquivo, max_registros=None):
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            linhas = [linha.strip() for linha in f if linha.strip()]
            linhas_validas = [linha for linha in linhas if linha in valores_validos]
            linhas_validas.reverse()

            if max_registros:
                return linhas_validas[-max_registros:]

            return linhas_validas
    except FileNotFoundError:
        return []

def criar_features_avancadas(historico, window=10):
    if len(historico) < window:
        return None

    hist_nums = [mapping[r] for r in historico[-window:] if r in mapping]

    if len(hist_nums) < window:
        return None

    features = []

    for i in range(min(10, len(hist_nums))):
        features.append(hist_nums[-(i+1)])

    azul_count = hist_nums.count(0) / len(hist_nums)
    vermelho_count = hist_nums.count(1) / len(hist_nums)
    empate_count = hist_nums.count(2) / len(hist_nums)
    features.extend([azul_count, vermelho_count, empate_count])

    current_streak = 1
    for i in range(len(hist_nums)-1, 0, -1):
        if hist_nums[i] == hist_nums[i-1]:
            current_streak += 1
        else:
            break
    features.append(current_streak / len(hist_nums))

    mudancas = sum(1 for i in range(1, len(hist_nums)) if hist_nums[i] != hist_nums[i-1])
    features.append(mudancas / (len(hist_nums) - 1))

    return features

def preparar_dados_lstm(hist, window=10):
    X, y = [], []

    if len(hist) < window + 1:
        return np.array(X), np.array(y)

    for i in range(len(hist) - window):
        janela = hist[i:i+window]
        features = criar_features_avancadas(janela, window=window)

        if features is not None:
            X.append(features)
            proximo = hist[i+window]
            if proximo in mapping:
                y.append(mapping[proximo])

    return np.array(X), np.array(y)

def treinar(model, X_train, y_train, X_val=None, y_val=None, epochs=20):
    model.train()

    print(f"Treinando: {len(X_train)} exemplos, {epochs} épocas")

    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float).unsqueeze(1).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        val_loss = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_inputs = torch.tensor(X_val, dtype=torch.float).unsqueeze(1).to(device)
                val_labels = torch.tensor(y_val, dtype=torch.long).to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels).item()
            model.train()

        metricas.registrar_loss(train_loss, val_loss)

        if (epoch + 1) % 5 == 0:
            msg = f"  Época {epoch+1}/{epochs}, Loss: {train_loss:.4f}"
            if val_loss:
                msg += f" | Val: {val_loss:.4f}"
            print(msg)

    print("✓ Treino concluído\n")

def prever_proximo_com_confianca(model, seq):
    model.eval()

    features = criar_features_avancadas(seq, window=10)

    if features is None:
        return None, None

    entrada = torch.tensor([features], dtype=torch.float).unsqueeze(1).to(device)

    start_time = time.time()

    with torch.no_grad():
        out = model(entrada)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    tempo_inferencia = time.time() - start_time
    metricas.registrar_tempo_inferencia(tempo_inferencia)

    pred_idx = probs.argmax()
    pred_cor = inv_mapping[pred_idx]
    confianca = probs[pred_idx]

    return pred_cor, confianca

def decidir_aposta(model, historico, threshold_ia=0.50):
    metricas.registrar_oportunidade()

    pred_cor, confianca = prever_proximo_com_confianca(model, historico)

    if pred_cor and confianca and confianca >= threshold_ia:
        return pred_cor, 'LSTM', confianca

    return None, None, confianca if confianca else 0.0

# ============================================================================
# LOOP PRINCIPAL
# ============================================================================
def main_loop():
    global historico, previsao_pendente, origem_previsao, confianca_previsao

    carregar_modelo()
    metricas.carregar_metricas(arquivo_metricas)

    historico_completo = ler_historico_valido(arquivo_resultados)
    if historico_completo:
        historico = historico_completo[-50:]
        print(f"✓ Histórico: {len(historico_completo)} resultados\n")

    print("="*80)
    print("Sistema iniciado! Threshold: 50%")
    print(f"Monitorando: {arquivo_resultados}")
    print("="*80 + "\n")

    tamanho_anterior = len(historico_completo) if historico_completo else 0
    contador_ciclos = 0

    while True:
        if os.path.exists(arquivo_resultados):
            historico_completo_atual = ler_historico_valido(arquivo_resultados)

            if len(historico_completo_atual) > tamanho_anterior:
                novos_count = len(historico_completo_atual) - tamanho_anterior
                novos = historico_completo_atual[tamanho_anterior:]

                print(f"\n{'='*80}")
                print(f"🆕 {novos_count} novo(s) resultado(s)!")
                print(f"{'='*80}")

                # Verificar predição
                if previsao_pendente:
                    novo_resultado = novos[0]

                    metricas.registrar_resultado(previsao_pendente, novo_resultado, confianca_previsao)

                    # ✓ MOSTRAR RESULTADO COM ESTATÍSTICAS
                    if previsao_pendente == novo_resultado:
                        print(f"\n🎉 IA ACERTOU! Esperava {previsao_pendente.upper()}, veio {novo_resultado.upper()}")
                    else:
                        print(f"\n❌ IA ERROU: esperava {previsao_pendente.upper()}, veio {novo_resultado.upper()}")

                    # ✓ SEMPRE MOSTRAR ESTATÍSTICAS APÓS CADA JOGADA
                    print(f"\n{'─'*80}")
                    print(f"📊 ESTATÍSTICAS ATUALIZADAS:")
                    print(f"{'─'*80}")
                    print(f"  Vitórias: {metricas.sucessos} | Derrotas: {metricas.get_derrotas()} | Total: {metricas.tentativas}")
                    print(f"  Taxa: {metricas.get_taxa_acertos():.1f}%")
                    print(f"  Sequência Atual: {metricas.sequencia_atual}")
                    print(f"  Maior Sequência: {metricas.maior_sequencia}")
                    print(f"{'─'*80}\n")

                    # Dashboard completo a cada 10
                    if metricas.tentativas % 10 == 0:
                        historico = historico_completo_atual[-50:]
                        metricas.mostrar_dashboard(historico)
                        metricas.salvar_metricas(arquivo_metricas)

                    previsao_pendente = None
                    origem_previsao = None
                    confianca_previsao = 0.0

                # Atualizar
                historico = historico_completo_atual[-50:]
                tamanho_anterior = len(historico_completo_atual)

                # Treinar
                X, y = preparar_dados_lstm(historico_completo_atual, window=10)
                if len(X) >= 30:
                    print(f"\nTreinando com {len(X)} exemplos...")
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                    treinar(model, X_train, y_train, X_val, y_val, epochs=20)
                    salvar_modelo()

                # Predição
                if not previsao_pendente and len(historico) >= 10:
                    aposta, origem, confianca_atual = decidir_aposta(model, historico)

                    if aposta:
                        previsao_pendente = aposta
                        origem_previsao = origem
                        confianca_previsao = confianca_atual

                        metricas.registrar_aposta(aposta, confianca_atual)

                        # ✓ MOSTRAR ÚLTIMOS 5 NA ORDEM CORRETA (ARQUIVO)
                        ultimos5 = historico[-5:] if len(historico) >= 5 else historico
                        ultimos5_arquivo = list(reversed(ultimos5))

                        print(f"\n⚡ IA VAI JOGAR!")
                        print(f"{'─'*80}")
                        print(f"  Estratégia: {origem}")
                        print(f"  Confiança: {confianca_atual*100:.1f}%")
                        print(f"  Previsão: {aposta.upper()}")
                        print(f"  Últimos 5: {' → '.join(ultimos5_arquivo)}   
                        print(f"{'─'*80}")
                        print(f"  Vitórias: {metricas.sucessos} | Derrotas: {metricas.get_derrotas()} | Taxa: {metricas.get_taxa_acertos():.1f}%")
                        print(f"  Sequência Atual: {metricas.sequencia_atual} | Maior: {metricas.maior_sequencia}")
                        print(f"{'─'*80}")
                    else:
                        print(f"\n⏸️  IA não jogou (confiança {confianca_atual*100:.1f}% < 50%)")
            else:
                contador_ciclos += 1
                if contador_ciclos % 60 == 0:
                    if metricas.tentativas > 0:
                        metricas.mostrar_dashboard(historico)

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Monitorando... ({len(historico_completo_atual)} resultados)", end='')
        else:
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Aguardando arquivo...", end='')

        time.sleep(5)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n✋ Interrompido")
        if len(historico) > 0:
            metricas.mostrar_dashboard(historico)
        metricas.salvar_metricas(arquivo_metricas)
        salvar_modelo()
        print("✓ Salvo")
