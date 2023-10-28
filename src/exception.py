import sys
import logging


def get_error_message_detail(error, error_detail):
    """
    This function is used to get the error message, file name and line number
    :param error:
    :param error_detail:
    :return:  formatted error message, file name and line number
    """
    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    error_message = f"Oops Something Went Bad: in {file_name} at line {line_number} with details {error}"
    return error_message


class CustomException(Exception):
    """
    This is a custom exception class that inherits from Exception class
    """

    def __init__(self, error_message, error_detail):
        """
        This is the constructor for the CustomException class
        :param error_message:
        :param error_detail:
        """
        super().__init__(error_message)
        self.error_message = get_error_message_detail(error=error_message, error_detail=error_detail)

    def __str__(self):
        """
        This is the string representation of the CustomException class
        :return:
        """
        return self.error_message


if __name__ == "__main__":
    try:
        a = 10/0
    except Exception as e:
        logging.info(f"Division by Zero Error")
        raise CustomException(error_message=e, error_detail=sys)


