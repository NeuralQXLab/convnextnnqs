import nqxpack

class SaveVariationalState:
    """
        Construct a callback which saves the variational state to file {file_prefix}vstate{step_number}.nk 
    
    """
    def __init__(self, save_every=1, file_prefix=""):
        self.save_every = save_every
        self.file_prefix = file_prefix
        self.i = 0
        self.last_saved=-1
        
    def __call__(self, step, log_data, driver):
        if (self.i - self.last_saved) >= self.save_every:
            vstate = driver.state
            print(f"Saving vstate to {self.file_prefix}vstate{step}.nk")
            nqxpack.save(vstate, f"{self.file_prefix}vstate{step}.nk")
            self.last_saved = self.i
        self.i += 1
        return True