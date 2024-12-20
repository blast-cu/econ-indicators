class Article(object):

    def __init__(self, id, headline, text, source, url,
                 is_econ, econ_sentences, econ_keywords, num_keywords,
                 date):
        self.id = id
        self.headline = headline
        self.text = text
        self.source = source
        self.url = url
        self.is_econ = is_econ
        self.econ_sentences = econ_sentences
        self.econ_keywords = econ_keywords
        self.num_keywords = num_keywords
        self.date = date

    def to_json(self):
        return {
            'id': self.id,
            'url': self.url,
            'source': self.source,
            'headline': self.headline,
            'text': self.text,
            'is_econ': self.is_econ,
            'econ_keywords': self.econ_keywords,
            'econ_sentences': self.econ_sentences,
            'num_keywords': self.num_keywords,
            'date': self.date
        }

    @staticmethod
    def from_json(dictionary):
        headline = dictionary['headline']
        if type(headline) is float:  # if headline is NaN from pd
            headline = None

        # if headline is None or too short, use generated headline
        if (headline is None or len(headline.split()) <= 4):
            if 'gen-headline' in dictionary:
                headline = dictionary['gen-headline']
            else:
                print("Invalid headline for article with id", dictionary['id'])

        if headline:  # remove padding tokens (for generated headlines)
            headline = headline.replace('<pad>', '')
            headline = headline.replace('</s>', '')
            headline = headline.strip()

        return Article(
            id=dictionary['id'],
            headline=headline,
            text=dictionary['text'],
            source=dictionary['source'],
            url=dictionary['url'],
            is_econ=dictionary['is_econ'],
            econ_sentences=dictionary['econ_sentences'],
            econ_keywords=dictionary['econ_keywords'],
            num_keywords=dictionary['num_keywords'],
            date=dictionary['date']
        )
