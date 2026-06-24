def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(ra, rb, result, k=32):
    ea = expected_score(ra, rb)

    new_ra = ra + k * (result - ea)
    new_rb = rb + k * ((1 - result) - (1 - ea))

    return new_ra, new_rb