class EloEngine:
    def __init__(self, teams, base=1500):
        self.ratings = {t: base for t in teams}

    def expected(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, home, away, result, k=20):
        # result: 1 = home win, 0.5 = draw, 0 = away win

        ra = self.ratings[home]
        rb = self.ratings[away]

        ea = self.expected(ra, rb)

        new_ra = ra + k * (result - ea)
        new_rb = rb + k * ((1 - result) - (1 - ea))

        self.ratings[home] = new_ra
        self.ratings[away] = new_rb

        return new_ra, new_rb