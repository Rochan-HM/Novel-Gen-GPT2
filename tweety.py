import twint

c = twint.Config()
c.Store_object = True


def get_tweets(search, limit):
    c.Search = search
    c.Limit = limit
    twint.run.Search(c)
    dat = twint.output.tweets_list
    res = []
    for t in dat:
        res.append(t.tweet)
    return res
