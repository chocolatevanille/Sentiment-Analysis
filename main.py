import PySimpleGUI as gui
import webbrowser
import re
from dotenv import load_dotenv
import os
import requests
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# load environment variables from the .env file
load_dotenv()

# access the environment variables
api_key = os.getenv('API_KEY')

gui.theme("DarkBlue")

# None -> None
# called when the Credits button is pressed
# it's me!
def display_credits():
    layout = [
        [gui.Text("Made by Noël Barron",justification='center')],
        [gui.Column([[gui.Button('GitHub',enable_events=True,key='-GITHUB-')]],
        justification='center')]
    ]
    window = gui.Window("Credits", layout, modal=True)
    url = 'https://github.com/chocolatevanille/PantheonofHallownestTracker'
    while True:
        event, values = window.read()
        if event == '-GITHUB-':
            webbrowser.open(url)
        elif event == gui.WIN_CLOSED:
            window.close()
            return None

# get_video_id: str -> str or bool
# obtains the video id from a YT link
def get_video_id(link):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, link)
    if match:
        return match.group(1)
    else:
        return False

# get_sentiment: str -> None
# called when the user inputs a link
def get_sentiment(link):
    #first, clean YouTube link
    video_id = get_video_id(link)
    if not video_id:
        return None
    max_results = 100  # Set max results per page (max is 100)
    pages = 0

    # Initialize variables
    comments = []
    next_page_token = None

    max_results = 100
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"

    while True:
        # Construct API request URL
        comments_url = f"{base_url}?part=snippet&videoId={video_id}&key={api_key}&maxResults={max_results}&pageToken={next_page_token if next_page_token is not None else ''}"

        # Make API request
        comments_response = requests.get(comments_url)
        comments_data = comments_response.json()

        # Extract comments from the response
        for item in comments_data["items"]:
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comment_text = str(comment_text)
            comment_text = re.sub(r'http\S+', '', comment_text) # remove URLs
            comment_text = re.sub(r'[^a-zA-Z\s]', '', comment_text, re.I|re.A) # remove numbers and special characters
            comment_text = comment_text.lower()
            comments.append(comment_text)

        # Check if there are more pages to fetch
        if 'nextPageToken' in comments_data:
            next_page_token = comments_data['nextPageToken']
        else:
            break  # No more pages, exit loop
        pages += 1
        if pages >= 30:
            break

    print(f"Number of comments: {len(comments)}")


    # load model and tokenizer
    model = BertForSequenceClassification.from_pretrained('sentiment_model')
    tokenizer = BertTokenizer.from_pretrained('sentiment_model')

    # testing with one comment
    # comment = comments[0]
    # print(comment)

    # tokenize single comment
    # inputs = tokenizer(comment, return_tensors='pt')

    # tokenize all comments
    # Tokenize input texts
    inputs = tokenizer.batch_encode_plus(
       comments,
       return_tensors='pt',  # return PyTorch tensors
       padding=True,         # pad to the maximum length in the batch
       truncation=True       # truncate comments longer than the maximum length
    )

    # Forward pass and prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = logits.argmax(dim=1).tolist()

    # Map predicted class to sentiment label
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    predicted_sentiments = [sentiment_labels[pred] for pred in predicted_classes]

    sentiment_tally = [0,0,0,0,0]
    for sentiment in predicted_sentiments:
        ind = sentiment_labels.index(sentiment)
        sentiment_tally[ind] = sentiment_tally[ind] + 1

    # get total number of sentiments
    # there is a faster way to do this (number of reviews), implement later
    sentiment_count = 0
    for i in sentiment_tally:
        sentiment_count += i
    print(f"Total number of reviews: {sentiment_count}")
    
    sentiment_percentage = []
    for i in sentiment_tally:
        sentiment_percentage.append(i*100/sentiment_count)
    print(f"Sentiment Percentages: {sentiment_percentage}")
    
    bar_layout = [
        [gui.Graph(canvas_size=(500, 400), graph_bottom_left=(0, 0), graph_top_right=(500, 400), key='-GRAPH-')],
        [gui.Button('Close',key='-CLOSE-SENTIMENTS-')]
    ]
    sentiments_window = gui.Window('Sentiment Analysis', bar_layout, finalize=True)
    graph = sentiments_window['-GRAPH-']

    for i, value in enumerate(sentiment_percentage):
        graph.draw_rectangle(top_left=(i*75 + 75, value * 3.5 + 50),bottom_right=(i*75 + 125, 50),fill_color='Purple')

    #bar_width = 50
    #padding = 20
    #for i, (sentiment, value) in enumerate(zip(sentiment_labels, sentiment_tally)):
    #    graph.draw_rectangle(top_left=(i * (bar_width + padding) + padding, 300 - value * 5),
    #                        bottom_right=((i + 1) * bar_width + i * padding, 300),
    #                        fill_color='blue')
    #    graph.draw_text(sentiment, (i * (bar_width + padding) + padding + 10, 280), color='black')
    while True:
        event, values = sentiments_window.read()
        if event == gui.WINDOW_CLOSED or event == '-CLOSE-SENTIMENTS-':
            break
    sentiments_window.close()

# designing the main window layout
layout_top = [  [gui.Text("YouTube Sentiment Analysis")]]
            
layout_middle = [gui.Text("Link:"),
                 gui.Input(default_text="https://www.youtube.com/watch?v=YbJOTdZBX1g",size=(100,12),tooltip="Your link goes here.",border_width=5,focus=True,key='-LINK-')
]

layout_bottom = [gui.Column([[gui.Button('Go!',key='-GO-',button_color="Green")]]),
                 gui.Column([[gui.Button('Credits',key='-CREDITS-')]]),
                 gui.Column([[gui.Button('Close',key='-CLOSE-')]])
]

layout = [
    [   
        layout_top,
        layout_middle,
        layout_bottom
    ]
]

# main window creation
window = gui.Window('YT Sentiment Analysis', layout,size=(750,540),element_justification='center',resizable=True,return_keyboard_events=True)

# event loop
def win_run():
    while True:
        event, values = window.read()
        if event == gui.WIN_CLOSED or event == '-CLOSE-': # when window closes
            break
        elif event == '-GO-' or event == '\r':
            link = values['-LINK-']
            get_sentiment(link)
        elif event == '-CREDITS-': # when the user wants to see who made the program
            display_credits() # hey it's me! wait didn't i already do that jo-
            continue
    window.close()


def main():
    print()
    win_run()


if __name__ == "__main__":
    main()