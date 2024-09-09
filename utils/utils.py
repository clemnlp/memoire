import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, NonlinearConstraint
import seaborn as sns


def format_datetime_columns(df, column_names=["DateOfBirth", "DateIn", "DateOut"]):
    for col in column_names:
        df[col] = pd.to_datetime(df[col])
    return df


def reframe_to_observation(df, year):
    beginning_year_timestamp = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end_year_timestamp = pd.Timestamp(year=year, month=12, day=31, hour=0)
    df.loc[df["DateOut"] > end_year_timestamp, "Status"] = False
    df["DateIn"] = df["DateIn"].apply(lambda x: max(x, beginning_year_timestamp))
    df["DateOut"] = df["DateOut"].apply(lambda x: min(x, end_year_timestamp))

    predicate = df["DateIn"] <= df["DateOut"]

    return df.loc[predicate]


def save_matrix(mat, path, ages=range(35, 65 + 1), years=range(2015, 2022 + 1)):
    df = pd.DataFrame(
        mat, columns=[f"Age {i}" for i in ages], index=[f"Year {i}" for i in years]
    )
    df.to_csv(path, sep=",", index=True, encoding="utf-8")

def load_matrix(filepath, ages=range(35,65+1), years=range(2015,2022+1), transpose=False):
    df = pd.read_csv(filepath, index_col=0)
    if transpose:
        df = df.T
    df.columns = [int(col) if isinstance(col, int) or col.isdigit() else int(col[4:]) for col in df.columns]
    df.index = [int(idx) if isinstance(idx, int) or idx.isdigit() else int(idx[5:]) for idx in df.index]
    # Find common years and ages
    common_years = df.index.intersection(years)
    common_ages = df.columns.intersection(ages)
    # Filter dataframes to only include common years and ages
    mat = df.loc[common_years, common_ages].to_numpy()
    return mat


def compute_ages(df):
    df["AgeIn"] = (df["DateIn"] - df["DateOfBirth"]) / pd.Timedelta(days=365.25)
    df["AgeOut"] = (df["DateOut"] - df["DateOfBirth"]) / pd.Timedelta(days=365.25)
    return df


def compute_death_expo(df, ages):
    n = len(ages)
    expo = np.zeros(n)
    death = np.zeros(n)

    for i in range(n):
        exposed_predicate = (df["AgeIn"] <= ages[i]) & (df["AgeOut"] >= ages[i])
        exposed = df.loc[exposed_predicate]
        death[i] = sum(exposed["Status"])
        expo[i] = exposed.shape[0]

        assert expo[i] >= death[i]

    return expo, death


def compute_death_expo_range(
    input_filename, ages=range(35, 65 + 1), years=range(2015, 2022 + 1)
):
    expo_list = []
    death_list = []

    df = pd.read_csv(input_filename)
    df = format_datetime_columns(df)
    df["Status"] = df["Status"] == "deceased"

    for year in years:
        reframed_df = df.copy(deep=True)
        reframed_df = reframe_to_observation(reframed_df, year)
        reframed_df = compute_ages(reframed_df)
        expo, death = compute_death_expo(reframed_df, ages)
        expo_list.append(expo)
        death_list.append(death)

    return np.vstack(death_list), np.vstack(expo_list)


def plot_matrix(X, Y, Z, title="Taux de mortalité", save=None):
    X, Y = np.meshgrid(X, Y)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surface = ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none")

    # Add a color bar which maps values to colors
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.1)

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Age", labelpad=15)
    ax.set_ylabel("Année", labelpad=15)
    ax.set_zlabel("Taux", labelpad=15)

    ax.tick_params(axis="z", labelsize=8, pad=5)

    ax.view_init(elev=40, azim=120)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    # Show the plot
    plt.show()


def normalize_matrix(matrix, method="min-max", range_min=0, range_max=1):
    """
    Normalizes a NumPy matrix using the specified method.

    Parameters:
    - matrix (np.ndarray): The matrix to normalize.
    - method (str): The normalization method, one of 'min-max', 'z-score', or 'l2'.
    - range_min (float): Minimum value of the range for Min-Max normalization. Default is 0.
    - range_max (float): Maximum value of the range for Min-Max normalization. Default is 1.

    Returns:
    - np.ndarray: The normalized matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix must be a NumPy array.")

    if method == "min-max":
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = range_min + (matrix - min_val) * (range_max - range_min) / (
            max_val - min_val
        )

    elif method == "z-score":
        mean_val = np.mean(matrix)
        std_dev_val = np.std(matrix)
        normalized_matrix = (matrix - mean_val) / std_dev_val

    elif method == "l2":
        l2_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / l2_norms

    else:
        raise ValueError(
            "Unknown normalization method. Use 'min-max', 'z-score', or 'l2'."
        )

    return normalized_matrix


def thatcher(x, alpha, beta, gamma):
    v_x = 1 + alpha * np.exp(beta * x)
    v_x_plus_one = 1 + alpha * np.exp(beta * (x + 1))

    result = 1 - np.exp(-gamma) * np.power(v_x / v_x_plus_one, 1 / beta)
    return result


def bongaarts(x, t, a, b, beta, gamma, lin=False):
    if lin:
        alpha = a * t + b
    else:
        alpha = np.exp(a * t + b)

    return thatcher(x, alpha, beta, gamma)


def mean_squared_error(matrix1, matrix2):
    # Ensure the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    # Compute the squared differences
    squared_diff = (matrix1 - matrix2) ** 2

    # Compute the mean of the squared differences
    mse = np.mean(squared_diff)

    return mse


def mean_absolute_error(matrix1, matrix2, weights=None):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    absolute_diff = np.abs(matrix1 - matrix2)
    if weights is not None:
        weights = normalize_matrix(weights)
        mae = np.mean(weights * absolute_diff)
    else:
        mae = np.mean(absolute_diff)

    return mae


def bongaarts_matrix(
    a, b, beta, gamma, lin=False, X=range(35, 65 + 1), T=range(2015, 2022 + 1)
):
    X, T = np.meshgrid(X, T)
    result_matrix = bongaarts(X, T, a, b, beta, gamma, lin=lin)

    return result_matrix


def distance(
    mat, a, b, beta, gamma, lin=False, X=range(35, 65 + 1), T=range(2015, 2022 + 1)
):
    result_matrix = bongaarts_matrix(a, b, beta, gamma, lin=lin, X=X, T=T)

    return mean_absolute_error(mat, result_matrix)


def linear_regression(x, y, plot=False, exp=False, title="Linear Regression"):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    if exp:
        y = np.log(y)

    # Number of data points
    n = len(x)

    # Calculate the sums needed for the formulas
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x**2)
    sum_xy = np.sum(x * y)

    # Calculate the slope (a) and intercept (b)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b = (sum_y - a * sum_x) / n

    if plot:
        # Generate the predicted y values for the linear regression line
        if exp:
            y_pred = np.exp(a * x + b)
        else:
            y_pred = a * x + b

        # Plot the original data points and the linear regression line
        plt.scatter(x, np.exp(y) if exp else y, color="blue", label="Data points")
        plt.plot(x, y_pred, color="red", label="Linear regression line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.show()

    return a, b


def makeham_estimator(x, q_x, plot=False, title="makeham estimator"):
    x = np.array(x)
    q_x = np.array(q_x)

    differences = np.diff(q_x)
    positive_y = np.log(differences[differences > 0])
    positive_x = x[:-1][differences > 0]

    slope, intercept = linear_regression(positive_x, positive_y, plot=plot, title=title)

    beta = slope
    alpha = np.exp(intercept) * beta * np.power((np.exp(beta) - 1), -2)
    gamma = np.mean(
        -np.log(1 - q_x) - (alpha / beta) * np.exp(beta * x) * (np.exp(beta) - 1)
    )

    params = {"beta": beta, "alpha": alpha, "gamma": gamma}

    return params


def thatcher_distance(params, qbrut, expo, x):
    """
    Parameters to optimize = alpha, beta, gamma
    """
    alpha, beta, gamma = params
    q_thatcher = thatcher(x, alpha, beta, gamma)

    return mean_absolute_error(qbrut, q_thatcher, weights=expo)


def thatcher_distance2(alpha, beta, gamma, qbrut, expo, x):
    """
    Parameters to optimize = alpha
    """
    q_thatcher = thatcher(x, alpha, beta, gamma)

    return mean_absolute_error(qbrut, q_thatcher, weights=expo)


def thatcher_per_year_objective_builder(mat_obj, x):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 10e-7, 10e-4, log=True)
        beta = trial.suggest_float("beta", 10e-3, 1, log=True)
        gamma = trial.suggest_float("gamma", -10e-4, -10e-7)

        try:
            result = thatcher_distance([alpha, beta, gamma], mat_obj, x)
            if np.isnan(result) or result is None:
                print("NAN", alpha, beta, gamma)
                return float("inf")
            else:
                return result
        except (OverflowError, ValueError, ZeroDivisionError):
            print(alpha, beta, gamma)
            return float("inf")

    return objective


def per_year_thatcher_params(
    death,
    expo,
    ages=range(35, 65 + 1),
    years=range(2015, 2022 + 1),
    projected_ages=range(0, 110 + 1),
    optimization_method="SLSQP",
    plot=False,
    save_path=None,
    filename=None,
):
    QBrut = death / expo
    years = np.array(years)
    x = np.array(ages)
    projected_x = np.array(projected_ages)

    alpha_array = np.zeros(len(years))
    beta_array = np.zeros(len(years))
    gamma_array = np.zeros(len(years))
    QThatcher = np.zeros((len(years), len(x)))

    bounds = ((10e-7, 10e-4), (10e-3, 1), (-10e-4, -10e-7))
    for year_idx, q_brut in enumerate(QBrut):
        initial_guess = makeham_estimator(
            x,
            q_brut,
            plot=plot,
            title=f"Makeham parameters estimation (Year {years[year_idx]})",
        )
        initial_guess = [
            initial_guess["alpha"],
            initial_guess["beta"],
            initial_guess["gamma"],
        ]

        constraint = NonlinearConstraint(
            lambda params: thatcher(projected_x, params[0], params[1], params[2]),
            np.zeros(projected_x.shape),
            np.inf,
        )
        result = minimize(
            thatcher_distance,
            initial_guess,
            args=(q_brut, expo[year_idx], x),
            method=optimization_method,
            constraints=[constraint],
            bounds=bounds,
        )

        alpha, beta, gamma = result.x

        alpha_array[year_idx] = alpha
        beta_array[year_idx] = beta
        gamma_array[year_idx] = gamma

        q_thatcher = thatcher(x, result.x[0], result.x[1], result.x[2])
        QThatcher[year_idx] = q_thatcher

        if plot:
            plt.scatter(x, q_brut, color="blue", label="Taux bruts")
            plt.plot(
                x,
                q_thatcher,
                color="red",
                label=f"Taux Thatcher ({alpha:.2E}, {beta:.2E}, {gamma:.2E})",
            )
            # plt.plot(x, q_thatcher_opt, color='green', label=f'Taux Thatcher (TPE: {study.best_params["alpha"]:.2E}, {study.best_params["beta"]:2E}, {study.best_params["gamma"]:.2E})')

            # plt.annotate(f'p-value: {pvalue_Thatcher:.4e}', xy=(0.05, 0.95), xycoords='axes fraction',
            # fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Linear Regression")
            plt.legend()
            plt.show()

    if plot:
        n_years = len(years)
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
            ax.plot(ages, QThatcher[year_idx], color="red", label="Modèle ajusté")
            ax.set_xlabel("Age")
            ax.set_title(f"Année {year}")
            if year_idx == 0:
                ax.set_ylabel("Taux")
            if year_idx == n_years - 1:
                ax.legend()
        # Hide any unused subplots
        for ax in axes[n_years:]:
            ax.axis("off")

        fig.suptitle(
            f"Taux {filename} (modèle ajusté par la méthode de Planchet-Kamega)"
        )
        # Adjust layout
        plt.tight_layout()

        plt.savefig(f"images/PLANCHET_{filename}_ajustement.png", bbox_inches="tight")
        plt.show()

        plt.scatter(years, alpha_array, color="blue", label="alpha")
        plt.xlabel("Year")
        plt.ylabel("alphas")
        plt.legend()
        plt.show()

        plt.scatter(years, beta_array, color="blue", label="beta")
        plt.xlabel("Year")
        plt.ylabel("betas")
        plt.legend()
        plt.show()

        plt.scatter(years, gamma_array, color="blue", label="gamma")
        plt.xlabel("Year")
        plt.ylabel("gammas")
        plt.legend()
        plt.show()

        plot_matrix(x, years, QThatcher, title="Thatcher per year")

    if save_path is not None:
        save_matrix(QThatcher, save_path, ages=ages, years=years)

    return {
        "alpha": alpha_array,
        "beta": beta_array,
        "gamma": gamma_array,
        "q": QThatcher,
    }


def per_year_thatcher_adjust_alpha(
    death,
    expo,
    alpha_array,
    beta_array,
    gamma_array,
    ages=range(35, 65 + 1),
    years=range(2015, 2022 + 1),
    projected_ages=range(0, 110 + 1),
    optimization_method="SLSQP",
    plot=False,
    save_path=None,
    filename=None,
):
    QBrut = death / expo
    years = np.array(years)
    x = np.array(ages)
    projected_x = np.array(projected_ages)

    beta_hat = np.mean(beta_array)
    gamma_hat = np.mean(gamma_array)

    new_alpha_array = np.zeros(len(years))
    QBongaarts = np.zeros((len(years), len(x)))

    bounds = (10e-7, 10e-4)
    for year_idx, q_brut in enumerate(QBrut):
        initial_guess = alpha_array[year_idx]

        constraint = NonlinearConstraint(
            lambda params: thatcher(projected_x, params, beta_hat, gamma_hat),
            np.zeros(projected_x.shape),
            np.inf,
        )
        result = minimize(
            thatcher_distance2,
            initial_guess,
            args=(beta_hat, gamma_hat, q_brut, expo[year_idx], x),
            method=optimization_method,
            constraints=[constraint],
            bounds=None,
        )
        alpha = result.x[0]
        q_thatcher = thatcher(x, result.x[0], beta_hat, gamma_hat)
        QBongaarts[year_idx] = q_thatcher

        new_alpha_array[year_idx] = alpha

        if plot:
            plt.scatter(x, q_brut, color="blue", label="Taux bruts")
            plt.plot(x, q_thatcher, color="red", label="Taux Thatcher")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Linear Regression")
            plt.legend()
            plt.show()

    if plot:
        n_years = len(years)
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
            ax.plot(
                ages,
                QBongaarts[year_idx],
                color="red",
                label="Modèle ajusté après avis d'expert",
            )
            ax.set_xlabel("Age")
            ax.set_title(f"Année {year}")
            if year_idx == 0:
                ax.set_ylabel("Taux")
            if year_idx == n_years - 1:
                ax.legend()
        # Hide any unused subplots
        for ax in axes[n_years:]:
            ax.axis("off")

        fig.suptitle(
            f"Taux {filename} (modèle ajusté par la méthode de Planchet-Kamega après avis d'expert)"
        )
        # Adjust layout
        plt.tight_layout()

        plt.savefig(
            f"images/PLANCHET_{filename}_ajustement_expert.png", bbox_inches="tight"
        )
        plt.show()
        plt.scatter(years, new_alpha_array, color="blue", label="alpha")
        plt.xlabel("Year")
        plt.ylabel("New alphas")
        plt.legend()
        plt.show()

        plot_matrix(x, years, QBongaarts, title="Thatcher per year (alpha ajusted)")

    if save_path is not None:
        save_matrix(QBongaarts, save_path, ages=ages, years=years)

    return {"alpha": alpha_array, "beta": beta_hat, "gamma": gamma_hat, "q": QBongaarts}


def bongaarts_projection(
    death,
    expo,
    alpha_array,
    beta,
    gamma,
    ages=range(35, 65 + 1),
    years=range(2015, 2022 + 1),
    projection_ages=range(0, 110 + 1),
    projection_years=range(2015, 2050 + 1),
    linear=False,
    plot=False,
    save_path=None,
):
    QBrut = death / expo
    years = np.array(years)
    x = np.array(ages)
    projection_years = np.array(projection_years)
    projection_x = np.array(projection_ages)

    QBongaarts_projection = np.zeros((len(projection_years), len(projection_x)))

    if linear:  # linear
        a, b = linear_regression(
            years, alpha_array, plot=plot, title="Linear regression: alpha = ax+b"
        )
    else:  # exponential
        a, b = linear_regression(
            years,
            alpha_array,
            plot=plot,
            exp=True,
            title="Linear regression: log(alpha) = ax+b",
        )

    for year_idx, year in enumerate(projection_years):
        if linear:
            alpha = a * year + b
        else:
            alpha = np.exp(a * year + b)

        q_bongaarts = thatcher(projection_x, alpha, beta, gamma)
        QBongaarts_projection[year_idx] = q_bongaarts

    if plot:
        plot_matrix(
            projection_x, projection_years, QBongaarts_projection, title="Projection"
        )

    if save_path is not None:
        save_matrix(
            QBongaarts_projection, save_path, ages=projection_x, years=projection_years
        )

    if linear:
        alpha = np.exp(a * projection_years + b)
    else:
        alpha = a * projection_years + b

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "q": QBongaarts_projection,
        "a": a,
        "b": b,
        "linear": linear,
    }


def planchet_projection(
    name,
    ages=range(35, 65 + 1),
    years=range(2015, 2022 + 1),
    projection_ages=range(0, 110 + 1),
    projection_years=range(2015, 2030 + 1),
    save=False,
):
    death, expo = compute_death_expo_range(f"./data/{name}.csv")
    if save:
        save_matrix(death / expo, f"./matrices/PLANCHET_{name}_brut.csv", ages, years)
    plot_matrix(ages, years, death / expo, title=f"Taux bruts {name}")

    per_year_thatcher_dict = per_year_thatcher_params(
        death,
        expo,
        ages=ages,
        years=years,
        optimization_method="SLSQP",
        plot=True,
        filename=name[6:],
    )

    per_year_thatcher_adjusted_alpha_dict = per_year_thatcher_adjust_alpha(
        death,
        expo,
        per_year_thatcher_dict["alpha"],
        per_year_thatcher_dict["beta"],
        per_year_thatcher_dict["gamma"],
        ages=ages,
        years=years,
        optimization_method="SLSQP",
        plot=True,
        filename=name[6:],
    )

    bongaarts_projection_dict = bongaarts_projection(
        death,
        expo,
        per_year_thatcher_adjusted_alpha_dict["alpha"],
        per_year_thatcher_adjusted_alpha_dict["beta"],
        per_year_thatcher_adjusted_alpha_dict["gamma"],
        ages=ages,
        years=years,
        projection_ages=projection_ages,
        projection_years=projection_years,
        linear=False,
        plot=True,
        save_path=f"./matrices/PLANCHET_{name}_projection.csv" if save else None,
    )

    plt.figure(figsize=(150, 30))
    sns.heatmap(
        bongaarts_projection_dict["q"],
        annot=True,
        cmap="plasma",
        linecolor="black",
        linewidths=1,
        fmt=".3e",
    )
    plt.show()

    print("ALPHAS")
    print(per_year_thatcher_dict["alpha"])
    print(per_year_thatcher_adjusted_alpha_dict["alpha"])
    print(bongaarts_projection_dict["alpha"])

    print("BETAS")
    print(per_year_thatcher_dict["beta"])
    print(per_year_thatcher_adjusted_alpha_dict["beta"])
    print(bongaarts_projection_dict["beta"])

    print("GAMMAS")
    print(per_year_thatcher_dict["gamma"])
    print(per_year_thatcher_adjusted_alpha_dict["gamma"])
    print(bongaarts_projection_dict["gamma"])

    print("A")
    print(bongaarts_projection_dict["a"])
    print("B")
    print(bongaarts_projection_dict["b"])
    print("LINEAR")
    print(bongaarts_projection_dict["linear"])
