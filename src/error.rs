use thiserror::Error;
// use smartstring::alias::String;

#[derive(Error, Debug)]
pub enum SmootError {
    #[error("Incompatible unit types {0} and {1}")]
    IncompatibleUnitTypes(String, String),

    #[error("Invalid operation {0} between units {1} and {2}")]
    InvalidOperation(&'static str, String, String),

    //==================================================
    // Parsing errors
    //==================================================
    #[error("line:{0} {1}")]
    ConflictingDefiniition(usize, String),

    #[error("line:{0} Invalid operator {1}")]
    InvalidOperator(usize, String),

    #[error("line:{0} Unexpected dimensions {1}")]
    UnexpectedDimension(usize, String),

    #[error("line:{0} Invalid unit expression {1}")]
    InvalidUnitExpression(usize, String),

    #[error("line:{0} Invalid quantity expression {1}")]
    InvalidQuantityExpression(usize, String),

    #[error("line:{0} Unknown unit {1}")]
    UnknownUnit(usize, String),

    #[error("Failed to load cache file {0}")]
    FailedToLoadCache(&'static str),

    #[error("Failed to decode cache file {0}")]
    FailedToDecodeCache(&'static str),

    #[error("Failed to write cache file {0}")]
    FailedToWriteCache(&'static str),
}

pub type SmootResult<T> = Result<T, SmootError>;
