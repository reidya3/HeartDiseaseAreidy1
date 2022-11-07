"""Module contains custom exceptions"""

class InvalidTxtError(Exception):
    """
    This exception is thrown when an
        invalid path of the model pickle file is supplied.
    """
    
class EmptyFieldError(Exception):
    """
    This exception is thrown when an user does not supply
        all the features, required by the model
    """