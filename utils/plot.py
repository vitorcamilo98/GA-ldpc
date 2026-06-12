import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_convergencia(csv_path, output_path="resultados/convergencia.png"):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for tipo in sorted(df["tipo"].unique()):
        sub = df[df["tipo"] == tipo]
        media = sub.groupby("geracao")["bler"].mean()
        std = sub.groupby("geracao")["bler"].std().fillna(0)
        ax.plot(media.index, media.values, label=tipo)
        ax.fill_between(media.index, media.values - std, media.values + std, alpha=0.15)
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Melhor BLER")
    ax.set_title("Convergencia - BLER")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for tipo in sorted(df["tipop"].unique()):
        sub = df[df["tipo"] == tipo]
        media = sub.groupby("geracao")["fitness"].mean()
        ax.plot(media.index, media.values, label=tipo)
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Fitness")
    ax.set_title("Convergencia - Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")



def plot_convergencia_girth(csv_path,  output_path="resultados/convergencia_girtrh.png"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    for tipo in sorted(df["tipo"].unique()):
        sub = df[df["tipo"] == tipo]
        media = sub.groupby("geracao")["bler"].mean()
        std = sub.groupby("geracao")["bler"].std().fillna(0)
        plt.plot(media.index, media.values, label=tipo)
        plt.fill_between(media.index, media.values - std, media.values + std, alpha=0.15)

    plt.xlabel("Geracao")
    plt.ylabel("Girth")
    plt.title("Comvergencia - Girth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_bler_vs_snr(csv_path, output_path="resultados/bler_vs_snr.png"):
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(8, 6))
    for tipo in sorted(df["tipo"].unique()):
        sub = df[df["tipo"] == tipo]
        media = sub.groupby("snr")["bler"].mean()
        plt.semilogy(media.index, media.values, "o-", label=tipo, markersize=4)

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BLER")
    plt.title("BLER vs Eb/N0")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_girth_histogram(csv_path, output_path="resultados/girth_histogram.png"):
    df = pd.read_csv(csv_path)

    idx = df.groupby(["tipo", "execucao"])["geracao"].idxmax()
    final = df.loc[idx]

    tipos = sorted(final["tipo"].unique())
    girths_per_tipo = [final[final["tipo"] == t]["girth"].values for t in tipos]

    plt.figure(figsize=(8, 6))
    bins = np.arange(
        min(final["girth"]) - 0.5,
        max(final["girth"]) - 1.5,
        1,
    )

    plt.hist(girths_per_tipo, bins=bins, label=tipos, edgecolor="black", alpha=0.7)
    plt.xlabel("Girth")
    plt.ylabel("Contagem (execucoes)")
    plt.title("Distribuicao de Girth - Melhor Codigo por Execucao")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_bler_boxplot(csv_path, snr_target=2.0, output_path="resultados/bler_boxplot.png"):
    df = pd.read_csv(csv_path)

    snrs = df["snr"].unique()
    closest = snrs[np.argmin(np.abs(snrs - snr_target))]
    sub = df[df["snr"] == closest]
    
    tipos = sorted(sub["tipo"].unique())
    data = [sub[sub["tipo"] == t]["bler"].values for t in tipos]

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data, labels=tipos, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(tipos)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    plt.ylabel("BLER")
    plt.title(f"Robustez - BLER @ Eb/N0 = {closest:.1f} dB")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def gerar_tabela_comparativa(res_path, conv_path, snr_target=2.0, output_path="resultados/tabela_comparativa.csv"):
    df_res = pd.read_csv(res_path)
    df_conv = pd.read_csv(conv_path)

    idx = df_conv.groupby(["tipo", "execucao"])["geracao"].idxmax()
    final_conv = df_conv.loc[idx]

    snrs = df_res["snr"].unique()
    closest = snrs[np.argmin(np.abs(snrs - snr_target))]
    final_res = df_res[df_res["snr"] == closest]

    rows = []
    for tipo in sorted(final_conv["tipo"].unique()):
        sc = final_conv[final_conv["tipo"] == tipo]
        sr = final_res[final_conv["tipo"] == tipo]

        rows.append({
            "Estrategia": tipo,
            "BLER (media)": f"{sr['bler'].mean():.4f}",
            "BLER (std)": f"{sr['bler'].std():.4f}",
            "Girth (media)": f"{sr['girth'].mean():.1f}",
            "Girth (std)": f"{sr['girth'].std():.1f}",
            "Ciclos-6 (media)": f"{sr['n_c6'].mean():.1f}",
            "Ciclos-6 (std)": f"{sr['n_c6'].std():.1f}",
            "Fitness (media)": f"{sr['fitness'].mean():.4f}",
            "Tempo (s)": f"{sr['tempo'].mean():.1f}" if "tempo" in sr.columns else "N/A",
        })

        tab = pd.DataFrame(rows)
        tab.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(tab.to_string(index=False))
        return tab

