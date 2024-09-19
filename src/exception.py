import sys
from .logger import logging



def error_message_detail(error, error_details:sys):
    _,_,exe_tb = error_details.exc_info()
    file = exe_tb.tb_frame.f_code.co_filename
    line = exe_tb.tb_lineno
    err = str(error)
    error_message = f'Error Occured in {file} at line: {line} \n Error: {err}'
    return error_message

    
    
class customException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details=error_detail)
        
    def __str__(self) -> str:
        return self.error_message
    
    

