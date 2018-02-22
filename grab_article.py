from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import requests

headers = {'User-Agent':'Mozilla/5.0'}
# r = requests.get("https://www.theverge.com/2018/2/22/17035300/htc-u11-plus-review-android-smartphone", headers=headers)
# r  = requests.get("https://en.wikipedia.org/wiki/Adolf_Hitler", headers=headers)
# r  = requests.get("https://www.cnn.com/2018/02/21/politics/trump-listening-sessions-parkland-students/index.html", headers=headers)
# r  = requests.get("https://www.washingtonpost.com/news/post-politics/wp/2018/02/22/attacks-would-end-trump-expresses-support-for-raising-assault-rifle-age-to-21-presses-cases-for-arming-some-teachers/?hpid=hp_hp-top-table-main_pp-trump-841am%3Ahomepage%2Fstory&utm_term=.973c927a015f", headers=headers)
r  = requests.get("https://www.nytimes.com/2018/02/21/us/politics/trump-guns-school-shooting.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=a-lede-package-region&region=top-news&WT.nav=top-news", headers=headers)
data = r.text

soup = BeautifulSoup(data, "lxml")

text = ""

for p in soup.find_all('p'):
	text = text + '\n' + p.getText()



# text needs to be in an array
text = [text]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
# print(vector.shape)
# print(type(vector))
print(vector.toarray())