import tqdm

from common import transcribe_listeners

class CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value
        
    def update(self, n):
        super().update(n)
        self._current += n
        
        for l in transcribe_listeners:
            l['progress_bar'].value = self._current / self.total
            l['page'].update()

