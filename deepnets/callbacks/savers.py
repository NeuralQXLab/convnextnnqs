from deepnets.utils.serialize import save_variables

class SaveVariables:
    """
        Callback which saves the variational state variables
    """
    def __init__(self, save_every=1, file_prefix=""):
        self.save_every = save_every
        self.file_prefix = file_prefix
        self.i = 0
        self.last_saved=-1
        
    def __call__(self, step, log_data, driver):
        if (self.i - self.last_saved) >= self.save_every:
            vstate = driver.state
            print(f"Saving vstate variables to {self.file_prefix}vars{step}.mpack")
            save_variables(f"{self.file_prefix}vars{step}.mpack",vstate)
            self.last_saved = self.i
        self.i += 1
        return True