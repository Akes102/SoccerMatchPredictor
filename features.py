def team_form(df, team):
    last = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(10)

    points = 0
    goals_for = 0
    goals_against = 0

    for _, r in last.iterrows():

        if r["HomeTeam"] == team:
            goals_for += r["FTHG"]
            goals_against += r["FTAG"]

            if r["FTR"] == "H":
                points += 3
            elif r["FTR"] == "D":
                points += 1

        else:
            goals_for += r["FTAG"]
            goals_against += r["FTHG"]

            if r["FTR"] == "A":
                points += 3
            elif r["FTR"] == "D":
                points += 1

    games = max(len(last), 1)

    return [
        points / games,
        goals_for / games,
        goals_against / games,
        (goals_for - goals_against) / games
    ]