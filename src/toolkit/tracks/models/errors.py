

class InvalidTrackGeneration(Exception):

    def __init__(self, message, if_issue_persists="consider changing track generation inputs"):
        super().__init__(f"{message}. If this issue persists, {if_issue_persists}.")
