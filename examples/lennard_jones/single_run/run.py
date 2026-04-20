import logging                                                          
logging.basicConfig(                                                                    
    level=logging.INFO,                     # or DEBUG     
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",           
)                                                                                       
from jaxrens.cli.cli import main                         
main()