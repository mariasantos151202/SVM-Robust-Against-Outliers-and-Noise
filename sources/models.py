import logging
import numpy as np


class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()

    def get_params(self, deep=True):
        # This method is used to retrieve hyperparameters for re-instantiation or tuning.
        # Collects common hyperparameters for SVM models.
        params = {}
        if hasattr(self, 'C'):
            params['C'] = self.C
        if hasattr(self, 'kernel'):
            params['kernel'] = self.kernel
        if hasattr(self, 'tol'):
            params['tol'] = self.tol
        if hasattr(self, 'max_iter'):
            params['max_iter'] = self.max_iter
        if hasattr(self, 'max_iter'):
            params['noise_factor'] = self.noise_factor
        
        return params

# %%
class SVM_STANDARD(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100, noise_factor=None):
        """Support vector machines implementation using simplified SMO optimization.

        Parameters
        ----------
        C : float, default 1.0  parametro de regularização
        kernel : Kernel object
        tol : float , default 1e-3, critério de paragem
        max_iter : int, default 100
        """
        self.C = C# Parâmetro de Regularização: Controla o equilíbrio entre
                          # ter uma margem larga e classificar tudo bem.
        self.tol = tol       # Tolerância: Critério de paragem. Se os ajustes forem
                          # menores que isto, o modelo considera que convergiu.
        self.max_iter = max_iter # Número máximo de passagens pelos dados.

        self.kernel=kernel
        self.noise_factor=noise_factor
        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None): #faz uma matriz kernel com os dados
        """Fit: preparação dos dados para a otimização"""
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples): # aqui vai preencher a matriz com os pontos i e j
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

#Matematicamente: Se os vetores apontam para a mesma direção, o valor é alto (semelhantes). Se forem perpendiculares, o valor é zero.


    def _train(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            alpha_prev = np.copy(self.alpha) # guardamos os alpah anteriores para conseguirmos comparar!

            for j in range(self.n_samples):
                # Pick random i para otimizar em par com um ponto j
                i = self.random_index(j)
                # eta: Mede a semelhança entre os pontos. Se for >= 0, não conseguimos
                # optimizar este par, por isso saltamos.
                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue
                # L e H: São os limites (Bounds) para o alpha. O alpha não pode ser menor que 0 nem maior que C.
                L, H = self._find_bounds(i, j)

                # Error for current examples
                e_i, e_j = self._error(i), self._error(j)

                # Save old alphas
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                # Update alpha: fundamental para a aprendizagem
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)

                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (
                    alpha_jo - self.alpha[j]
                )


                # Find intercept: cálculo do Bias/Viés:
                b1 = (
                    self.b - e_i - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i]- self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                )
                b2 = (
                    self.b- e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j]- self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]

                )
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        logging.info("Convergence has reached after %s." % iters)

        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        """Error for single example."""


        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j):
        """Find L and H such that L <= alpha <= H.
        Also, alpha must satisfy the constraint 0 <= αlpha <= C.
        """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples)
        return i




# %%
class SVM_modified_3(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100, noise_factor=0):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.noise_factor=noise_factor # calculado no pre processing utilizando o DNA

        self.kernel=kernel

        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

    def _train(self):
        self._run_smo(adaptive_C=False)
        #NOVA ALTERAÇÃO
        if self.noise_factor> 0.05 and self.noise_factor< 0.15: #noise moderado
          self.C = self.C * 0.80
        else:
          if self.noise_factor>0.15: #noise alto
            self.C = self.C * 0.50

        # Calcular erros após 1ª passagem e ajustar C por ponto
        self.C_individual = np.array([
            self.C / (1.0 + abs(self._error(i)))
            for i in range(self.n_samples)
        ])

        self._run_smo(adaptive_C=True)

    def _run_smo(self, adaptive_C=False):
            iters = 0
            while iters < self.max_iter:
                iters += 1
                alpha_prev = np.copy(self.alpha)

                # --- VETORIZAÇÃO DOS ERROS ---
                # Calcula f(x) e erros para todas as amostras de uma vez
                predictions = np.dot(self.alpha * self.y, self.K) + self.b
                errors = predictions - self.y

                for i in range(self.n_samples):
                    Ci = self.C_individual[i] if adaptive_C else self.C
                    Ei = errors[i]
                    
                    """
                    Adicionamos a condição KKT à implementação base, de acordo com o resultado da derivação parcial da função L em relação 
                    ao vetor w e vetor b. 
                    """

                    # Verifica se o ponto i viola as condições KKT
                    if (self.y[i] * Ei < -self.tol and self.alpha[i] < Ci) or \
                    (self.y[i] * Ei > self.tol and self.alpha[i] > 0):

                        # Heurística de seleção para j (máxima diferença de erro)
                        mask = np.arange(self.n_samples) != i
                        j = np.where(mask)[0][np.argmax(np.abs(Ei - errors[mask]))]
                        
                        Ej = errors[j]
                        Cj = self.C_individual[j] if adaptive_C else self.C

                        # Cálculo do eta
                        eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                        if eta >= 0:
                            continue

                        L, H = self._find_bounds(i, j, Ci, Cj)
                        alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                        # Update alpha j
                        self.alpha[j] -= (self.y[j] * (Ei - Ej)) / eta
                        self.alpha[j] = self.clip(self.alpha[j], H, L)
                        
                        # Update alpha i
                        self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])
                        self.alpha[i] = self.clip(self.alpha[i], Ci, 0)

                        # ---- Backtrack se o erro piorou ----
                        if adaptive_C:
                            # Cálculo pontual para o backtrack
                            e_j_depois = (self.alpha * self.y) @ self.K[:, j] + self.b - self.y[j]
                            if abs(e_j_depois) > abs(Ej):
                                self.alpha[i], self.alpha[j] = alpha_io, alpha_jo
                                continue

                        # Update bias
                        b1 = (self.b - Ei - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i] - 
                            self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j])
                        b2 = (self.b - Ej - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] - 
                            self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j])

                        if 0 < self.alpha[i] < Ci:
                            self.b = b1
                        elif 0 < self.alpha[j] < Cj:
                            self.b = b2
                        else:
                            self.b = 0.5 * (b1 + b2)
                        
                        # Atualiza o vetor de erros localmente para a próxima iteração do loop i
                        predictions = np.dot(self.alpha * self.y, self.K) + self.b
                        errors = predictions - self.y

                    diff = np.linalg.norm(self.alpha - alpha_prev)
                    if diff < self.tol:
                        break

            logging.info("Convergence has reached after %s." % iters)
            self.sv_idx = np.where(self.alpha > 0)[0]

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j, Ci, Cj):
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(Cj, Ci - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - Ci)
            H = min(Cj, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):

        """
        Esta função foi corrigida. i = np.random.randint(0, self.n_samples -1 ). A fold não permitia a escolha de um dos índices como potencial vetor de suporte;
        Provoca a degernação do algoritmo de otimização. As condições KKT não são satisfeitas. O output nunca é gerado.
        """
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples)
        return i

class SVM_modified_3_ponto_1(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100, noise_factor=0):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.noise_factor=noise_factor # calculado no pre processing utilizando o DNA

        self.kernel=kernel

        self.b = 0
        self.alpha = None
        self.K = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
        self.alpha = np.zeros(self.n_samples)
        self.sv_idx = np.arange(0, self.n_samples)
        return self._train()

    def _train(self):
        self._run_smo(adaptive_C=False)
        #NOVA ALTERAÇÃO
        if self.noise_factor> 0.05 and self.noise_factor< 0.15: #noise moderado
          self.C = self.C * 0.80
        else:
          if self.noise_factor>0.15: #noise alto
            self.C = self.C * 0.50
        # podemos usar uma função para calcular o treshold dos valores limites de noise do dataset para definir o C 

        # Calcular erros após 1ª passagem e ajustar C por ponto
        self.C_individual = np.array([
            self.C / (1.0 + abs(self._error(i)))
            for i in range(self.n_samples)
        ])

        

        self._run_smo(adaptive_C=True)

    def _run_smo(self, adaptive_C=False):
            iters = 0
            while iters < self.max_iter:
                iters += 1
                alpha_prev = np.copy(self.alpha)

                # --- VETORIZAÇÃO DOS ERROS ---
                # Calcula f(x) e erros para todas as amostras de uma vez
                predictions = np.dot(self.alpha * self.y, self.K) + self.b
                errors = predictions - self.y

                for j in range(self.n_samples):
                    # --- HEURÍSTICA DE SELECÇÃO (Primeira Escolha i) ---
                    # Verifica se o ponto i viola as condições KKT
                    i = self.random_index(j)   #ESCOLHA RANDOM

                    Ci = self.C_individual[i] if adaptive_C else self.C
                    Ei = errors[i]


                    if (self.y[i] * Ei < -self.tol and self.alpha[i] < Ci) or \
                    (self.y[i] * Ei > self.tol and self.alpha[i] > 0):
                    
                                        
                        Ej = errors[j]
                        Cj = self.C_individual[j] if adaptive_C else self.C

                        # Cálculo do eta
                        eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                        if eta >= 0:
                            continue

                        L, H = self._find_bounds(i, j, Ci, Cj)
                        alpha_io, alpha_jo = self.alpha[i], self.alpha[j]

                        # Update alpha j
                        self.alpha[j] -= (self.y[j] * (Ei - Ej)) / eta
                        self.alpha[j] = self.clip(self.alpha[j], H, L)
                        
                        # Update alpha i
                        self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])
                        self.alpha[i] = self.clip(self.alpha[i], Ci, 0)

                        # ---- Backtrack se o erro piorou ----
                        if adaptive_C:
                            # Cálculo pontual para o backtrack
                            e_j_depois = (self.alpha * self.y) @ self.K[:, j] + self.b - self.y[j]
                            if abs(e_j_depois) > abs(Ej):
                                self.alpha[i], self.alpha[j] = alpha_io, alpha_jo
                                continue

                        # Update bias
                        b1 = (self.b - Ei - self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, i] - 
                            self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j])
                        b2 = (self.b - Ej - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] - 
                            self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j])

                        if 0 < self.alpha[i] < Ci:
                            self.b = b1
                        elif 0 < self.alpha[j] < Cj:
                            self.b = b2
                        else:
                            self.b = 0.5 * (b1 + b2)
                        
                        # Atualiza o vetor de erros localmente para a próxima iteração do loop i
                        predictions = np.dot(self.alpha * self.y, self.K) + self.b
                        errors = predictions - self.y

                    diff = np.linalg.norm(self.alpha - alpha_prev)
                    if diff < self.tol:
                        break

                logging.info("Convergence has reached after %s." % iters)
                self.sv_idx = np.where(self.alpha > 0)[0]

    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self._predict_row(X[i, :]))
        return result

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]

    def _find_bounds(self, i, j, Ci, Cj):
        """
        H--> limite superior
        L--> limite inferior
        L <= alpha <= H
           """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(Cj, Ci - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - Ci)
            H = min(Cj, self.alpha[i] + self.alpha[j])
        return L, H

    def random_index(self, z):
        i = z
        while i == z:
            i = np.random.randint(0, self.n_samples)
        return i













