import random

class KR_agileUtils:
    def __init__(self):
        print("[UTIL] Util class initialized")

    def rewriteFile(self, filename, text) -> bool:
        with open(filename,"w") as f:
            f.write(text)
        return 1
    @staticmethod
    def randomGenBalanced(ran=1):
        """
        Balanced Random Number Generator -> add to utils class in future
        """
        return (random.random() - 0.5) * ran