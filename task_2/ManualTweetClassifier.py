import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import messagebox
from tkinter import simpledialog

## USAGE: Set INPUT_FILE to the file to annotate, then run this file as the main module

INPUT_FILE = ""

pd.options.mode.chained_assignment = None 

class InputWindow(tk.Frame):
    def __init__(self, parent, input_f):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.tweets = pd.read_csv(input_f)

        parent.withdraw()
        output_append = simpledialog.askstring(title="Output Filename", prompt="Input the text to append to the output filename (close to save over input file):")
        if output_append != None:
            self.outputFilename = input_f[:-4] + "_" + output_append + ".csv"
        else:
            self.outputFilename = input_f
        
        if "tweet_sentiment" not in self.tweets:
            self.position = 0
            self.tweets["tweet_sentiment"] = ""
        else:
            foundRow = False
            for index, row in self.tweets.iterrows():
                if pd.isna(row["tweet_sentiment"]):
                    self.position = index
                    foundRow = True
                    break
                
            if not foundRow:
                messagebox.showinfo(message="All tweets already have a sentiment value.")
                self.parent.destroy()
                return

        parent.deiconify()
        
        self.text = tk.Label(parent, text=self.tweets["text"][self.position], font=(None, 20), wraplength=500)
        self.text.grid(row=1, column=1, pady=20)
        self.positive=tk.Button(parent, text="Positive", height=10, width=20, command=self.positiveSelected)
        self.positive.grid(row=0, column=0, padx=10)
        parent.bind("1", self.keyPositiveSelected)
        self.neutral=tk.Button(parent, text="Neutral", height=10, width=20, command=self.neutralSelected)
        self.neutral.grid(row=0, column=1, padx=10)
        parent.bind("2", self.keyNeutralSelected)
        self.negative=tk.Button(parent, text="Negative", height=10, width=20, command=self.negativeSelected)
        self.negative.grid(row=0, column=2, padx=10)
        parent.bind("3", self.keyNegativeSelected)


    def keyPositiveSelected(self, event):
        self.positiveSelected()
        
    def positiveSelected(self):
        col = self.tweets["tweet_sentiment"]
        col.iloc[self.position] = "positive"
        self.tweets["tweet_sentiment"] = col
        self.updatePosition()

    def keyNeutralSelected(self, event):
        self.neutralSelected()

    def neutralSelected(self):
        col = self.tweets["tweet_sentiment"]
        col.iloc[self.position] = "neutral"
        self.tweets["tweet_sentiment"] = col
        self.updatePosition()

    def keyNegativeSelected(self, event):
        self.negativeSelected()

    def negativeSelected(self):
        col = self.tweets["tweet_sentiment"]
        col.iloc[self.position] = "negative"
        self.tweets["tweet_sentiment"] = col
        self.updatePosition()

    def updatePosition(self):
        self.saveTweets()
        self.position += 1
        if self.position >= self.tweets.shape[0]:
            self.saveTweets()
            messagebox.showinfo(message="All tweets annotated.")
            self.parent.destroy()
        else:
            self.text.config(text=self.tweets["text"][self.position])

    def saveTweets(self):
        self.tweets.to_csv(self.outputFilename)
        

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Manual Tweet Classifier")
    root.deiconify()
    window = InputWindow(root, INPUT_FILE)
    root.mainloop()





