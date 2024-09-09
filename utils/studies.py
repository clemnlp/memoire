import numpy as np
import optuna
import matplotlib.pyplot as plt
from .utils import (
    bongaarts_matrix,
    mean_absolute_error,
    plot_matrix,
    save_matrix,
    compute_death_expo_range,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def final_bongaarts_objective_builder(mat_obj, ages, years):
    years = np.array(years)

    def objective(trial):
        a = trial.suggest_float("a", -0.1, -0.05)
        b = trial.suggest_float("b", 0, 500)
        beta = trial.suggest_float("beta", 10e-2, 0.3, log=True)
        gamma = trial.suggest_float("gamma", 10e-6, 10e-4, log=True)

        alpha = np.exp(a * years + b)

        try:
            QBongaarts = bongaarts_matrix(
                a, b, beta, -gamma, lin=False, X=ages, T=years
            )
            result = mean_absolute_error(QBongaarts, mat_obj)
            if np.isnan(result) or result is None:
                print("NAN", alpha, beta, -gamma)
                return float("inf")
            else:
                return result
        except (OverflowError, ValueError, ZeroDivisionError):
            print(alpha, beta, gamma)
            return float("inf")

    return objective


def bongaarts_study(
    name,
    ages=range(35, 65 + 1),
    years=range(2015, 2022 + 1),
    projection_ages=range(0, 110 + 1),
    projection_years=range(2015, 2050 + 1),
    objective_builder=final_bongaarts_objective_builder,
    n_trials=1000,
    plot=False,
    save=False,
):
    n_years = len(years)

    death, expo = compute_death_expo_range(f"./data/input_{name}.csv")
    QBrut = death / expo
    if plot:
        plot_matrix(ages, years, QBrut, title=f"Taux bruts {name}")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective_builder(QBrut, ages, years), n_trials=n_trials, show_progress_bar=True)
    QBongaarts = bongaarts_matrix(
        study.best_params["a"],
        study.best_params["b"],
        study.best_params["beta"],
        study.best_params["gamma"],
        lin=False,
        X=ages,
        T=years,
    )

    if plot:
        ncols = 3
        nrows = (n_years + ncols - 1) // ncols  # Compute the number of rows needed
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), sharey=True
        )

        # Flatten the axes array for easy iteration (in case of multiple rows)
        axes = axes.flatten()

        # Plotting
        for year_idx, (ax, year) in enumerate(zip(axes, years)):
            ax.scatter(ages, QBrut[year_idx], color="blue", label="Taux bruts")
            ax.plot(ages, QBongaarts[year_idx], color="red", label="Modèle ajusté")
            ax.set_xlabel("Age")
            ax.set_title(f"Année {year}")
            if year_idx == 0:
                ax.set_ylabel("Taux")
            if year_idx == n_years - 1:
                ax.legend()
        # Hide any unused subplots
        for ax in axes[n_years:]:
            ax.axis("off")

        fig.suptitle(f"Taux {name} (modèle ajusté par la méthode alternative)")
        # Adjust layout
        plt.tight_layout()
        if save:
            plt.savefig(f"images/OPTUNA_{name}_ajustement.png", bbox_inches="tight")
        plt.show()

    QBongaarts_projection_all_ages = bongaarts_matrix(
        study.best_params["a"],
        study.best_params["b"],
        study.best_params["beta"],
        study.best_params["gamma"],
        lin=False,
        X=projection_ages,
        T=projection_years,
    )

    QBongaarts_projection = bongaarts_matrix(
        study.best_params["a"],
        study.best_params["b"],
        study.best_params["beta"],
        study.best_params["gamma"],
        lin=False,
        X=ages,
        T=projection_years,
    )

    if plot:
        plot_matrix(
            projection_ages,
            projection_years,
            QBongaarts_projection_all_ages,
            title=f"Bongaarts {name}",
            save=f"./images/BONGAARTS_{name}_all.png" if save else None,
        )
        plot_matrix(
            ages,
            projection_years,
            QBongaarts_projection,
            title=f"Bongaarts {name}",
            save=f"./images/BONGAARTS_{name}.png" if save else None,
        )

    if save:
        save_matrix(QBrut, f"./matrices/BRUT_{name}.csv", ages=ages, years=years)
        save_matrix(death, f"./matrices/BRUT_DEATH_{name}.csv", ages=ages, years=years)
        save_matrix(expo, f"./matrices/BRUT_EXPO_{name}.csv", ages=ages, years=years)
        save_matrix(
            QBongaarts_projection_all_ages,
            f"./matrices/BONGAARTS_{name}_all.csv",
            ages=projection_ages,
            years=projection_years,
        )
        save_matrix(
            QBongaarts_projection,
            f"./matrices/BONGAARTS_{name}.csv",
            ages=ages,
            years=projection_years,
        )

    print(
        "Taux minimum sur la projection bongaarts:",
        np.min(QBongaarts_projection_all_ages),
    )

    return QBongaarts_projection_all_ages
