import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass

# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log tranformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            ) # log1p handles log(0) by calculating log(1+x)
        logging.info("log tranformation cmopleted.")    
        return df_transformed
    
# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self , features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """    
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features : {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standar scaling completed.")
        return df_transformed
    

# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features, feature_range=(0,1)):
       """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
       self.features = features 
       self.scaler = MinMaxScaler(feature_range=feature_range) 

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            "Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed

    
# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self,features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features   
        self.encoder = OneHotEncoder(sparse=False,drop="first")
    def apply_transformation(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        logging.info(df.info())
        logging.info(df.columns.to_list())
        for i in self.features:
            if i in df.columns:
                logging.info(f"{i} esta")
            else:
                logging.info(f"{i}  no esta")

        logging.info(df.columns.tolist())
        df_transformed = df.copy()
        logging.info("1",df_transformed.columns.to_list())

        encoded_df=pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        logging.info("2",encoded_df.columns.to_list())
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        logging.info("3",df_transformed.columns)
        df_transformed = pd.concat([df_transformed,encoded_df],axis=1)

        logging.info("One-hot encoding completed.")
        return df_transformed

# --- New strategy: Count Encoding, added to feature_engineering.py ---

class CountEncoding(FeatureEngineeringStrategy):
    """
    Encodes high-cardinality categorical features (e.g. CODIGO_DEL_PRODUCTO)
    by replacing each category with its frequency (count) in the training data.

    Like TargetEncoding, this strategy must be fit only on the training set
    and then reused (transform) on test/inference data, to avoid leaking
    information about how often a category appears in unseen data.
    """

    def __init__(self, features):
        """
        Parameters:
        features (list): categorical columns to count-encode
                          (e.g. ['CODIGO_DEL_PRODUCTO']).
        """
        self.features = features

        # Learned state: one frequency dictionary per feature.
        # Populated only after fit_transform() runs on the training set.
        self.counts_ = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Learns category frequencies from TRAINING data and applies them
        to that same df.

        Parameters:
        df (pd.DataFrame): training dataframe.

        Returns:
        pd.DataFrame: df with categorical features replaced by their counts.
        """
        logging.info(f"Fitting count encoding on training data for: {self.features}")
        df_transformed = df.copy()

        for feature in self.features:
            # value_counts() gives {category: frequency} directly.
            freq_map = df[feature].value_counts().to_dict()
            self.counts_[feature] = freq_map

            df_transformed[feature] = df[feature].map(freq_map)

        logging.info("Count encoding fit_transform completed.")
        return df_transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a PREVIOUSLY LEARNED frequency mapping (from fit_transform)
        to new data (test set or inference data). Unseen categories get a
        fallback count of 0.

        Parameters:
        df (pd.DataFrame): dataframe to transform (test set or new data).

        Returns:
        pd.DataFrame: df with categorical features replaced by their counts.
        """
        if not self.counts_:
            raise RuntimeError(
                "CountEncoding.transform() called before fit_transform(). "
                "Fit on training data first."
            )

        logging.info(f"Applying learned count encoding to: {self.features}")
        df_transformed = df.copy()

        for feature in self.features:
            mapping = self.counts_[feature]
            # Unseen products get count 0 — they've never been observed.
            df_transformed[feature] = df[feature].map(mapping).fillna(0)

        logging.info("Count encoding transform completed.")
        return df_transformed

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Satisfies the abstract base class contract. Delegates to
        fit_transform — should only be called on the TRAINING split.
        """
        return self.fit_transform(df)

# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEnginner: 
    def __init__(self,strategy:FeatureEngineeringStrategy) :
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy=strategy
    def set_strategy(self,strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy
    def apply_feature_engineering(self,df :pd.DataFrame)-> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy")
        return self._strategy.apply_transformation(df)
    

# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass
