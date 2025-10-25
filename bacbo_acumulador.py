from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from colorama import init, Fore, Style
import time
import os
from datetime import datetime
import hashlib

init(autoreset=True)

LINHAS = 5
COLUNAS = 10
ULTIMO_HASH = None
ULTIMOS_RESULTADOS = None
ARQUIVO_RESULTADOS = "bacbo_resultados.txt"
MAX_RESULTADOS = 100000
contador_capturas = 0
contador_salvamentos = 0

def colorir_cor(cor):
    if cor == 'azul':
        return Fore.BLUE + 'P' + Style.RESET_ALL
    if cor == 'vermelho':
        return Fore.RED + 'B' + Style.RESET_ALL
    if cor == 'Empate':
        return Fore.YELLOW + 'T' + Style.RESET_ALL
    return cor

def gerar_hash(lista):
    """Gera hash único para lista de resultados"""
    return hashlib.md5(','.join(lista).encode()).hexdigest()

def carregar_resultados():
    if not os.path.exists(ARQUIVO_RESULTADOS):
        return []
    with open(ARQUIVO_RESULTADOS, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f.readlines() if linha.strip()]

def salvar_resultados(resultados):
    if len(resultados) > MAX_RESULTADOS:
        resultados = resultados[:MAX_RESULTADOS]
    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(r + "\n")
    return len(resultados)

def validar(resultados):
    return all(r in {'azul', 'vermelho', 'Empate'} for r in resultados)

def capturar_grid(driver):
    try:
        container = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="latest-50-outcomes"]')
        divs = container.find_elements(By.CSS_SELECTOR, 'div.grid.grid-cols-10.gap-0.justify-center')

        resultados = []
        if len(divs) == 5:
            for div in divs:
                imgs = div.find_elements(By.TAG_NAME, 'img')
                if len(imgs) == 10:
                    for img in imgs:
                        src = img.get_attribute('src')
                        if 'B.png' in src:
                            resultados.append('vermelho')
                        elif 'P.png' in src:
                            resultados.append('azul')
                        elif 'TIE.png' in src:
                            resultados.append('Empate')
                else:
                    return None
        else:
            return None

        if len(resultados) == 50 and validar(resultados):
            return resultados
    except:
        return None

# Driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get('https://casinoscores.com/pt-br/bac-bo/')

try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="latest-50-outcomes"]'))
    )
except:
    driver.quit()
    exit()

# Primeira captura
print("🔄 Sincronizando...")
time.sleep(2)
resultados_site = capturar_grid(driver)

if not resultados_site:
    driver.quit()
    exit()

resultados_acumulados = carregar_resultados()

if len(resultados_acumulados) == 0:
    resultados_acumulados = resultados_site.copy()
    salvar_resultados(resultados_acumulados)

ULTIMO_HASH = gerar_hash(resultados_site)
ULTIMOS_RESULTADOS = resultados_site.copy()
contador_salvamentos += 1

print("✅ Pronto\n")

# LOOP
while True:
    try:
        resultados = capturar_grid(driver)

        if resultados:
            contador_capturas += 1
            hash_atual = gerar_hash(resultados)

            # Compara hash - se diferente, grid mudou!
            if hash_atual != ULTIMO_HASH:
                # Detecta QUANTOS valores mudaram
                novos_count = 0
                for i in range(50):
                    if resultados[i] != ULTIMOS_RESULTADOS[i]:
                        novos_count += 1
                    else:
                        # Quando encontra igual, para
                        break

                # Se detectou mudança mas novos_count é 0,
                # significa que houve deslocamento, adiciona pelo menos 1
                if novos_count == 0:
                    novos_count = 1

                novos = resultados[:novos_count]

                print(f"\n✅ MUDANÇA! {len(novos)} novo(s)")
                print(f"   Hash anterior: {ULTIMO_HASH[:8]}...")
                print(f"   Hash atual:    {hash_atual[:8]}...")
                print(f"   Novos: {', '.join(novos[:3])}" + ("..." if len(novos) > 3 else ""))

                for n in reversed(novos):
                    resultados_acumulados.insert(0, n)

                total = salvar_resultados(resultados_acumulados)
                contador_salvamentos += 1
                print(f"   Total: {total}/{MAX_RESULTADOS}")

                ULTIMO_HASH = hash_atual
                ULTIMOS_RESULTADOS = resultados.copy()

            # Grid
            grid = [['' for c in range(COLUNAS)] for l in range(LINHAS)]
            for idx, r in enumerate(resultados):
                linha = idx // COLUNAS
                coluna = idx % COLUNAS
                if linha < LINHAS:
                    grid[linha][coluna] = colorir_cor(r)

            os.system('cls' if os.name == 'nt' else 'clear')
            print("Bead Plate Bac Bo")
            print(f"Total: {len(resultados_acumulados)}/{MAX_RESULTADOS} | Capturas: {contador_capturas} | Salvamentos: {contador_salvamentos}")
            print("    " + " ".join([str(i+1).rjust(2) for i in range(COLUNAS)]))
            for l_idx, linha in enumerate(grid):
                print(chr(65 + l_idx), " ".join(c.rjust(6) if c else " " for c in linha))
        else:
            print("\r⏳ Aguardando...", end='', flush=True)

        time.sleep(2)

    except Exception as e:
        print(f"\n✗ Erro: {e}")
        time.sleep(2)

driver.quit()
