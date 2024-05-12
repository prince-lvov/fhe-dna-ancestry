from copy import deepcopy
import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from concrete.ml.sklearn.xgb import XGBClassifier
from concrete.ml.sklearn import LogisticRegression

from src.Base.base import Base
from src.Smooth.smooth import Smoother
from src.Smooth.utils import slide_window


class ConcreteLRBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_multithread = True
        self.init_base_models(
            lambda : LogisticRegression(penalty="l2", C=3., solver="liblinear", max_iter=1000)
        )

    def __get_log_iter(self, X):
        slide_window = np.lib.stride_tricks.sliding_window_view

        # pad
        if self.context != 0.0:
            X = self.pad(X)

        # convolve
        M_ = self.M + 2 * self.context
        idx = np.arange(0, self.C, self.M)[:-2]
        X_b = slide_window(X, M_, axis=1)[:, idx, :]

        # stack
        train_args = tuple(zip(self.models[:-1], np.swapaxes(X_b, 0, 1)))
        rem = self.C - self.M * self.W
        train_args += (
            (
                self.models[-1],
                X[:, X.shape[1] - (M_ + rem) :],
            ),
        )

        # compile
        return tqdm.tqdm(
            train_args,
            total=self.W,
            bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}",
            position=0,
            leave=True,
        )

    def compile(self, X):
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.compile_vectorized(X)
            except AttributeError:
                print(
                    "Vectorized implementation requires numpy versions 1.20+.. Using loopy version.."
                )
                self.vectorize = False

    def compile_base_model(self, b, X):
        return b.compile(X)

    def compile_vectorized(self, X):
        for b in self.__get_log_iter(X):
            self.compile_base_model(*b)

    def predict_proba(self, X, fhe="disable"):
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.predict_vectorized(X, fhe=fhe)
            except AttributeError:
                print(
                    "Vectorized implementation requires numpy versions 1.20+.. Using loopy version.."
                )
                self.vectorize = False

    def predict_proba_base_model(self, b, X, fhe="disable"):
        return b.predict_proba(X, fhe=fhe)

    def predict_vectorized(self, X, fhe="disable"):
        B = np.array(
            [self.predict_proba_base_model(*b, fhe=fhe) for b in self.__get_log_iter(X)]
        )
        return np.swapaxes(B, 0, 1)
        

class ConcreteSmoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnofix = True
        assert self.W >= 2*self.S, "Smoother size to large for given window size. "
        self.model = XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, reg_lambda=1, reg_alpha=0,
            n_jobs=self.n_jobs, random_state=self.seed,
            use_label_encoder=False, objective='multi:softprob',
        )

    def process_base_proba(self, B, y=None):
        B_slide, y_slide = slide_window(B, self.S, y)
        return B_slide, y_slide
    
    def compile(self, B):
        self.model.compile(B)

    def predict_proba(self, B, fhe="disable"):
        B_s, _ = self.process_base_proba(B)
        proba = self.model.predict_proba(B_s, fhe=fhe)
        return proba.reshape(-1, self.W, self.A)


class ConcreteGnomix():

    def __init__(self, C, M, A, S,
                snp_pos=None, snp_ref=None, snp_alt=None, population_order=None, missing_encoding=2, # dataset specific, TODO: store in one object
                n_jobs=None, path=None, # configs
                calibrate=False, context_ratio=0.5, mode_filter=False, # hyperparams
                seed=None, verbose=False
    ):
        """
        Inputs
           C: chromosome length in SNPs
           M: number of windows for chromosome segmentation
           A: number of ancestry considered
        """

        self.C = C
        self.M = M
        self.A = A
        self.S = S
        self.W = self.C//self.M # number of windows

        # configs
        self.path = path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # data
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.snp_alt = snp_alt
        self.population_order = population_order

        # gnomix hyperparams
        self.context = int(self.M*context_ratio)
        self.calibrate = calibrate

        self.base = ConcreteLRBase(chm_len=self.C, window_size=self.M, num_ancestry=self.A,
                            missing_encoding=missing_encoding, context=self.context,
                            n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)

        self.smooth = ConcreteSmoother(n_windows=self.W, num_ancestry=self.A, smooth_window_size=self.S,
                            n_jobs=self.n_jobs, calibrate=self.calibrate, mode_filter=mode_filter, 
                            seed=self.seed, verbose=self.verbose)
        
        # model stats
        self.time = {}
        self.accuracies = {}

        # gen map df
        self.gen_map_df = {}

    def write_gen_map_df(self,gen_map_df):
        self.gen_map_df = deepcopy(gen_map_df)

    def conf_matrix(self, y, y_pred):

        cm = confusion_matrix(y.reshape(-1), y_pred.reshape(-1))
        indices = sorted(np.unique( np.concatenate((y.reshape(-1),y_pred.reshape(-1))) ))

        return cm, indices

    def train(self, data, retrain_base=True, evaluate=True, compile=True):
        (X_t1,y_t1), (X_t2,y_t2), (X_v,y_v) = data

        print("Training base model...")
        self.base.train(X_t1, y_t1)

        print("Training smoother...")
        B_t2 = self.base.predict_proba(X_t2)
        self.smooth.train(B_t2, y_t2)

        if evaluate:
            print("Evaluating model...")

            Acc = {}
            CM  = {}

            print ("training accuracy")
            B_t1 = self.base.predict_proba(X_t1)

            y_t1_pred = np.concatenate(
                [self.smooth.predict(_) for _ in np.array_split(B_t1, 5)]
            )
            y_t2_pred = self.smooth.predict(B_t2)

            Acc["base_train_acc"],   Acc["base_train_acc_bal"]   = self.base.evaluate(X=None,   y=y_t1, B=B_t1)
            Acc["smooth_train_acc"], Acc["smooth_train_acc_bal"] = self.smooth.evaluate(B=None, y=y_t2, y_pred=y_t2_pred)
            CM["train"] = self.conf_matrix(y=y_t1, y_pred=y_t1_pred)
            
            print ("val accuracy")
            if X_v is not None:
                B_v = self.base.predict_proba(X_v)
                y_v_pred  = self.smooth.predict(B_v)
                Acc["base_val_acc"],     Acc["base_val_acc_bal"]     = self.base.evaluate(X=None,   y=y_v,  B=B_v )
                Acc["smooth_val_acc"],   Acc["smooth_val_acc_bal"]   = self.smooth.evaluate(B=None, y=y_v,  y_pred=y_v_pred )
                CM["val"] = self.conf_matrix(y=y_v, y_pred=y_v_pred)

            self.accuracies = Acc
            self.Confusion_Matrices = CM

        if retrain_base:
            if X_v is not None:
                X_t, y_t = np.concatenate([X_t1, X_t2, X_v]), np.concatenate([y_t1, y_t2, y_v])
            else:
                X_t, y_t = np.concatenate([X_t1, X_t2]), np.concatenate([y_t1, y_t2])
            del X_t1, X_t2, X_v, y_t1, y_t2, y_v

            print("Re-training base models...")
            self.base.train(X_t, y_t)

        if compile:
            print ("Base model compile")
            self.base.compile(X_t)
            print ("Smooth model compile")
            B_t, _ = self.smooth.process_base_proba(B_t2)
            self.smooth.compile(B_t)