use thiserror::Error;

#[derive(Error, Debug)]
pub enum SmootError {
    #[error("Incompatible unit types {0} and {1}")]
    IncompatibleUnitTypes(String, String),

    #[error("{0}")]
    InvalidOperation(String),

    #[error("{0}")]
    NoSuchElement(String),

    //==================================================
    // Parsing errors
    //==================================================
    #[error("{0}")]
    DimensionError(String),

    #[error("{0}")]
    ParseTreeError(String),

    #[error("{0}")]
    ExpressionError(String),

    #[error("{0}")]
    CacheError(String),

    #[error("{0}")]
    FileError(String),

    //==================================================
    // Array errors
    //==================================================
    #[error("Invalid array dimensionality {0}")]
    InvalidArrayDimensionality(String),

    #[error("Mismatched shape {0}")]
    MismatchedArrayShape(String),
}

pub type SmootResult<T> = Result<T, SmootError>;
