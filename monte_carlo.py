# ============================================================================
# This code contains functions to perform the Monte Carlo test on MSSA results + to plot the significance 
# author: artemis zr
# started: 21/07/2026
# ============================================================================

import numpy as np 
from statsmodels.tsa.ar_model import AutoReg
import sys

sys.path.append('/Users/azr/Desktop/MSSA/scripts/')
from preprocessing_embedding import embedding


def select_ar_order(x, max_lags=10):
    """
    Select best AR order (with AIC method)
    """
    best_aic = np.inf
    best_p = 1
    for p in range(1, max_lags + 1):
        try:
            res = AutoReg(x, lags=p, old_names=False).fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_p = p
        except Exception:
            continue
    return best_p


def generate_arp_surrogate(X, n=1, max_lags=10, verbose=True):
    """
    Génère n réalisations surrogates AR(p) qui préservent
    la structure d'autocorrélation de chaque colonne de X.
    L'ordre p est sélectionné automatiquement par AIC.

    Parameters
    ----------
    X        : np.ndarray (N, D)
    n        : int — nombre de surrogates à générer
    max_lags : int — ordre AR maximum testé pour l'AIC
    verbose  : bool

    Returns
    -------
    surrogates : np.ndarray (n, N, D)
    ar_orders  : list[int] — ordre retenu par colonne
    """
    N, D = X.shape
    surrogates = np.zeros((n, N, D))
    ar_orders = []

    for d in range(D):
        x = X[:, d]
        mu = x.mean()
        xc = x - mu

        p = select_ar_order(xc, max_lags=max_lags)
        ar_orders.append(p)
        if verbose:
            print(f"  Colonne {d}: ordre AR retenu = {p}")

        res = AutoReg(xc, lags=p, old_names=False).fit()
        params = res.params
        const  = params[0]
        phi    = params[1:]
        sigma  = np.sqrt(res.sigma2)

        for k in range(n):
            noise = np.random.normal(0, sigma, N)
            s = np.zeros(N)
            s[:p] = xc[:p]

            for t in range(p, N):
                s[t] = const + np.dot(phi, s[t-1::-1][:p]) + noise[t]  

            surrogates[k, :, d] = s + mu

    return surrogates, ar_orders


def monte_carlo_mssa(X, M, eigvect_original, eigval_original, U_original,
                     n_surrogates=100, percentile=95,
                     max_lags=10, type='ECON'):
    """
    U_original : np.ndarray (N-M+1, K) — vecteurs singuliers gauches
                 retournés par mssa_workflow (BKi uniquement).
                 Passer None si type BK ou eig (utilise eigvect à la place).
    """
    N, D = X.shape
    K = len(eigval_original)
    LAMBDA_mc_AR = np.zeros((K, n_surrogates))

    if type == 'ECON':
        type = 'BKi' if (N - M + 1) < (D * M) else 'BK'
    print(f"Type : {type}")

    # Détermine le vecteur de projection à utiliser
    use_U = (type == 'BKi') and (U_original is not None)
    if use_U:
        print("Projection Allen & Robertson via U (espace réduit N-M+1)")
    else:
        print("Projection Allen & Robertson via eigvect (espace D*M)")

    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1
    X_scaled = X / X_std

    print("Ajustement AR(p) et génération des surrogates...")
    surrogates_scaled, ar_orders = generate_arp_surrogate(
        X_scaled, n=n_surrogates, max_lags=max_lags, verbose=True
    )
    print(f"Ordres AR retenus : {ar_orders}")
    surrogates = surrogates_scaled * X_std

    for i in range(n_surrogates):
        Xs     = surrogates[i]
        Xs_emb = embedding(Xs, M)                              # (N-M+1, D*M)
        Xs_emb_s = Xs_emb / np.sqrt(N - M + 1)                # normalisation identique à mssa_workflow

        if use_U:
            # Projection dans l'espace réduit (N-M+1) via U
            C_reduit = Xs_emb_s @ Xs_emb_s.T                  # (N-M+1, N-M+1)
            AR_proj  = U_original.T @ C_reduit @ U_original    # (K, K)
        else:
            # Projection dans l'espace complet (D*M) via eigvect
            C_full  = Xs_emb_s.T @ Xs_emb_s                   # (D*M, D*M)
            AR_proj = eigvect_original.T @ C_full @ eigvect_original  # (K, K)

        LAMBDA_mc_AR[:, i] = np.diag(AR_proj)[:K]

        if i % 20 == 0:
            print(f"  Surrogate {i+1}/{n_surrogates}...")

    lo = (100 - percentile) / 2
    hi = percentile + lo
    p_low  = np.percentile(LAMBDA_mc_AR, lo, axis=1)
    p_high = np.percentile(LAMBDA_mc_AR, hi, axis=1)
    is_significant = eigval_original[:K] > p_high

    return p_low, p_high, is_significant



def plot_mc_significance(eigval_obs, expvar_obs, surr_low, surr_high,
                         is_significant, freq_dom, M, varset_name, percentile,
                         n_modes_plot=30, save_path=None):
    """
    Deux panneaux :
    - gauche  : spectre par INDEX, avec enveloppe Monte Carlo
    - droite  : spectre par FRÉQUENCE, modes significatifs en couleur
    """
    n = min(n_modes_plot, len(eigval_obs))
    idx = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"MSSA — Test Monte Carlo (M={M}, {n_modes_plot} premiers modes)", fontsize=13)

    # ---- Panneau gauche : par index ----
    ax = axes[0]
    ax.fill_between(idx, surr_low[:n], surr_high[:n],
                    color='lightblue', alpha=0.6, label=f'Enveloppe MC ({percentile}%)')
    ax.plot(idx, surr_high[:n], 'b--', linewidth=0.8)
    ax.plot(idx, surr_low[:n],  'b--', linewidth=0.8)
    
    # Points : rouge si significatif, gris sinon
    colors = ['#E74C3C' if s else '#888888' for s in is_significant[:n]]
    ax.scatter(idx, eigval_obs[:n], c=colors, zorder=5, s=40)
    ax.plot(idx, eigval_obs[:n], 'k-', linewidth=0.7, alpha=0.5)
    
    ax.set_xlabel("Index du mode")
    ax.set_ylabel("Valeur propre")
    ax.set_title("Spectre par index")
    ax.legend(fontsize=9)

    # ---- Panneau droite : par fréquence ----
    ax = axes[1]
    freq = freq_dom[:n]
    ev   = eigval_obs[:n]
    sig  = is_significant[:n]
    sh   = surr_high[:n]

    # Errorbars (enveloppe MC résumée par le seuil haut uniquement ici)
    ax.axhline(y=np.median(surr_high[:n]), color='blue', linestyle='--',
               linewidth=0.8, alpha=0.7, label='Médiane seuil MC')

    for i in range(n):
        c = '#E74C3C' if sig[i] else '#AAAAAA'
        ax.plot([freq[i], freq[i]], [0, ev[i]], color=c, linewidth=1.2)
        ax.scatter(freq[i], ev[i], color=c, s=40, zorder=5)
        if sig[i]:
            ax.annotate(f"{i+1}\n({round(1/freq[i],1)}a)" if freq[i] > 0 else f"{i+1}",
                        (freq[i], ev[i]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5, color='#C0392B')

    ax.set_xlabel("Fréquence (cycles/an)")
    ax.set_ylabel("Valeur propre")
    ax.set_title("Spectre par fréquence (rouge = significatif)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()