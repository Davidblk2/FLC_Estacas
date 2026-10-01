import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io 

# ==========================================
# 1. MOTOR DE CÁLCULO (MOLAS DO SOLO - TIETZ)
# ==========================================
def calcular_molas_tietz(df_solo, diametro_cm):
    res = df_solo.copy()
    diametro_m = diametro_cm / 100.0
    res["Cz (tf/m³)"] = res["Coeficiente m (tf/m⁴)"] * res["Profundidade Z (m)"]
    res["Kx (kN/m)"] = res["Cz (tf/m³)"] * 10.0 * diametro_m
    res["Kx (kN/m)"] = np.round(res["Kx (kN/m)"], 0)
    return res

# ==========================================
# 2. MOTOR DE CÁLCULO ESTRUTURAL (MEF)
# ==========================================
def calcular_esforcos_estaca(L, D_m, E_concreto, molas_kx, H, M):
    n_elements = int(L)
    n_nodes = n_elements + 1
    ndof = 2 * n_nodes
    
    K = np.zeros((ndof, ndof))
    F_vec = np.zeros(ndof)
    I = (np.pi * (D_m)**4) / 64 
    
    for i in range(n_elements):
        le = 1.0 
        ke = (E_concreto * I / le**3) * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])
        idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for r in range(4):
            for c in range(4):
                K[idx[r], idx[c]] += ke[r, c]
                
    for i in range(n_nodes):
        idx_mola = min(i, n_elements - 1)
        K[2*i, 2*i] += molas_kx[idx_mola]
        
    # Esforços no topo
    F_vec[0] = H  
    F_vec[1] = -M 
    
    U = np.linalg.solve(K, F_vec)
    deslocamentos = U[0::2]
    momentos = np.zeros(n_nodes)
    
    for i in range(n_elements):
        le = 1.0
        ke = (E_concreto * I / le**3) * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])
        idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
        u_e = U[idx]
        f_e = ke @ u_e
        momentos[i] = abs(-f_e[1])
        if i == n_elements - 1:
            momentos[i+1] = abs(f_e[3])
            
    return deslocamentos, momentos

# ==========================================
# 3. INTERFACE DO USUÁRIO (FRONT-END)
# ==========================================
st.set_page_config(page_title="Estaca Isolada - Tietz", layout="wide")
st.title("🏗️ Dimensionamento de Estaca Isolada sob Flexão")
st.write("Análise pelo Método de Waldemar Tietz (Interação Solo-Estrutura).")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Esforços no Topo da Estaca")
    nk = st.number_input("Esforço Normal - Nk (kN):", value=51.73, step=1.0)
    mk = st.number_input("Momento Fletor - Mk (kN.m):", value=95.50, step=1.0)
    fk = st.number_input("Esforço Horizontal - Fk (kN):", value=14.00, step=1.0)

with col2:
    st.header("2. Geometria da Estaca")
    diametro_cm = st.number_input("Diâmetro da Estaca (cm):", value=35.0, step=1.0)
    comprimento = st.number_input("Comprimento Total (m):", value=6, step=1)
    fck = st.number_input("Fck do Concreto (MPa):", value=30.0, step=5.0)

st.divider()
st.header("3. Perfil do Solo e Coeficientes")
col_ref, col_edit = st.columns([1, 1.5])

with col_ref:
    st.write("**Tabela de Consulta: Valores de 'm' (Waldemar Tietz)**")
    tabela_referencia = pd.DataFrame({
        "Consistência (Argiloso)": ["Meio líquido (Turfa)", "Muito mole", "Mole", "Média", "Rija", "Muito rija", "Dura"],
        "SPT Base": ["0", "1", "3", "6", "12", "22", "> 30"],
        "m (tf/m⁴)": [25, 75, 150, 300, 500, 700, 900]
    })
    st.dataframe(tabela_referencia, hide_index=True)

with col_edit:
    st.write("**Edite o Perfil do Solo da Obra**")
    
    comp_int = int(comprimento)
    # Valores de referência baseados na memória de cálculo
    spt_base = [2, 2, 14, 15, 21, 18]
    m_base = [25, 25, 540, 560, 660, 620]
    
    # Ajusta automaticamente o tamanho das listas para igualar ao comprimento
    spt_lista = (spt_base + [0] * comp_int)[:comp_int]
    m_lista = (m_base + [25] * comp_int)[:comp_int]
    
    dados_iniciais = pd.DataFrame({
        "Profundidade Z (m)": range(1, comp_int + 1),
        "N_SPT": spt_lista,
        "Coeficiente m (tf/m⁴)": m_lista
    })
    df_editavel = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)

# ==========================================
# 4. AÇÃO DO BOTÃO E GERAÇÃO DO RELATÓRIO
# ==========================================
if st.button("Gerar Cálculo e Prancha", type="primary"):
    
    df_molas = calcular_molas_tietz(df_editavel, diametro_cm)
    molas_kx = df_molas["Kx (kN/m)"].values
    
    E_cs = 26000000.0 # kN/m2 (C30)
    D_m = diametro_cm / 100.0
    desloc, momentos = calcular_esforcos_estaca(comprimento, D_m, E_cs, molas_kx, fk, mk)
    
    mom_max = max(momentos)
    desloc_max = max(abs(desloc * 1000))
    
    st.divider()
    st.header("📊 Memória de Cálculo - Resultados")
    
    # --- CONSTRUÇÃO DA IMAGEM ---
    fig = plt.figure(figsize=(20, 6)) 
    
    # ----------------------------------------------------
    # Subplot 1: Modelo Estrutural
    # ----------------------------------------------------
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot([0, 0], [0, -comprimento], color='#8b5a2b', linewidth=4) 
    
    for z in range(1, int(comprimento) + 1):
        kx = molas_kx[z-1]
        y_mola = -z
        ax1.plot([-0.5, -0.3, -0.1, 0], [y_mola, y_mola+0.1, y_mola-0.1, y_mola], color='#2ca02c', lw=2)
        ax1.text(-0.6, y_mola, f"{int(kx)} kN/m", color='#2ca02c', fontsize=10, va='center', ha='right', fontweight='bold')
        
    ax1.plot([-0.3, 0.3], [-comprimento, -comprimento], color='black', lw=2)
    for i in np.linspace(-0.3, 0.3, 6):
        ax1.plot([i, i-0.15], [-comprimento, -comprimento-0.2], color='black', lw=1.5)

    # Força Horizontal Fk 
    ax1.annotate("", xy=(0, 0), xytext=(-2.2, 0), arrowprops=dict(facecolor='blue', edgecolor='blue', width=2, headwidth=8))
    ax1.text(-1.1, 0.15, f"{fk} kN", color='blue', fontsize=12, ha='center', va='bottom', fontweight='bold')
    
    # Força Vertical Nk 
    ax1.annotate("", xy=(0, 0), xytext=(0, 2.0), arrowprops=dict(facecolor='blue', edgecolor='blue', width=2, headwidth=8))
    ax1.text(-0.2, 1.0, f"{nk} kN", color='blue', fontsize=12, ha='right', va='center', fontweight='bold')
    
    # Momento Fletor Mk (Menor, mais fino e posicionado lateralmente)
    arc = mpatches.FancyArrowPatch((0.2, 0.6), (0.7, 0.0), connectionstyle="arc3,rad=-0.5", color="blue", lw=1.5, mutation_scale=12)
    ax1.add_patch(arc)
    ax1.text(0.9, 0.3, f"{mk} kN.m", color='blue', fontsize=12, fontweight='bold', ha='left', va='center')

    # Ajuste dos limites para não cortar os textos
    ax1.set_xlim(-3.5, 2.5)
    ax1.set_ylim(-comprimento - 0.5, 2.5)
    ax1.axis('off')
    ax1.set_title("Modelo Estrutural", fontweight='bold', family='Arial', fontsize=13)

    # ----------------------------------------------------
    # Subplot 2: Deslocamentos
    # ----------------------------------------------------
    ax2 = fig.add_subplot(1, 4, 2)
    profundidades = np.arange(0, comprimento + 1, 1)
    ax2.plot(desloc * 1000, -profundidades, marker='o', color='blue', linewidth=2)
    ax2.set_title("Deslocamento (mm)", fontweight='bold', family='Arial')
    ax2.set_ylabel("Profundidade (m)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axvline(0, color='black', linewidth=1)
    
    # ----------------------------------------------------
    # Subplot 3: Momentos Fletores
    # ----------------------------------------------------
    ax3 = fig.add_subplot(1, 4, 3, sharey=ax2)
    ax3.plot(momentos, -profundidades, marker='s', color='red', linewidth=2)
    ax3.fill_betweenx(-profundidades, 0, momentos, color='red', alpha=0.3)
    ax3.set_title("Momento Fletor (kN.m)", fontweight='bold', family='Arial')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.axvline(0, color='black', linewidth=1)
    
    # ----------------------------------------------------
    # Subplot 4: Texto do Resumo
    # ----------------------------------------------------
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis('off')
    
    y = 0.95
    linha = 0.08
    def texto(x, y_pos, txt, negrito=False, cor='black', tam=12):
        peso = 'bold' if negrito else 'normal'
        ax4.text(x, y_pos, txt, fontsize=tam, family='Arial', weight=peso, color=cor, transform=ax4.transAxes)

    texto(0.05, y, "RESUMO GERAL", negrito=True, tam=14)
    y -= linha * 1.5
    texto(0.05, y, "[1] DADOS DO PROJETO", negrito=True)
    y -= linha
    texto(0.05, y, f"Diâmetro da Estaca: {diametro_cm} cm")
    y -= linha
    texto(0.05, y, f"Comprimento: {comprimento} m")
    y -= linha * 1.5
    texto(0.05, y, "[2] ESFORÇOS APLICADOS", negrito=True)
    y -= linha
    texto(0.05, y, f"Carga Vertical (Nk): {nk:.2f} kN")
    y -= linha
    texto(0.05, y, f"Momento Fletor (Mk): {mk:.2f} kN.m")
    y -= linha
    texto(0.05, y, f"Esforço Horizontal (Fk): {fk:.2f} kN")
    y -= linha * 1.5
    texto(0.05, y, "[3] RESULTADOS MÁXIMOS", negrito=True)
    y -= linha
    texto(0.05, y, f"Deslocamento Topo: {desloc_max:.2f} mm", cor='blue', negrito=True)
    y -= linha
    texto(0.05, y, f"Momento (Msd): {mom_max:.2f} kN.m", cor='red', negrito=True)

    st.pyplot(fig)
    
    # Botão de Download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300) 
    buf.seek(0)
    
    st.download_button(
        label="💾 Baixar Prancha Completa (PNG)",
        data=buf,
        file_name="relatorio_estaca_flexao.png",
        mime="image/png"
    )