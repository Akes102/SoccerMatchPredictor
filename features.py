def team_form(df, team):
    matches = df[
        (df["HomeTeam"] == team) |
        (df["AwayTeam"] == team)
    ].tail(10)

    wins = draws = losses = 0
    gf = ga = 0

    for _, r in matches.iterrows():

        if r["HomeTeam"] == team:
            gf += r["FTHG"]
            ga += r["FTAG"]

            if r["FTR"] == "H":
                wins += 1
            elif r["FTR"] == "D":
                draws += 1
            else:
                losses += 1

        else:
            gf += r["FTAG"]
            ga += r["FTHG"]

            if r["FTR"] == "A":
                wins += 1
            elif r["FTR"] == "D":
                draws += 1
            else:
                losses += 1

    played = max(len(matches), 1)

    return [
        wins / played,
        draws / played,
        losses / played,
        gf / played,
        ga / played,
        (gf - ga) / played
    ]