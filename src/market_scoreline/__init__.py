"""market_scoreline — predict match scorelines purely from Pinnacle's closing line.

Pipeline:  fetch (this) -> devig -> inversion (market -> lambdas) -> matrix -> predict.
Bets are placed ~45 min before kickoff, so the captured line is the *closing* line.
"""
