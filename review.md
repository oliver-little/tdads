# David's Code Review
## The summary

In summary, you should probably read the Python style guide (PEP-8) to increase general readability, in particular the sections about naming functions and variables, and spacing around operators. You should also read the section of the Google style guide on docstrings to document your code a little better. I've linked these below. Let me know if there's anything you want clarification on.

## The review
Your functions could do with some docstrings - information about what a function takes as parameters and returns - preferably in [Google format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (or at least a consistent format)

On line 26, the ... notation is kind of hard to read. I'll admit this is a personal preference and what you have isn't technically bad, I just think it's nice to know how big your dimensions are rather than selecting everything. 

Your comments are generally pretty good, (could do with some docstrings) but e.g. when you import tf you don't need to put a comment there. Just a nitpick really, feel free to ignore this one. Also, some of your contents could use some more detail, for example why are you summing images and rows on line 48? Could potentially use a comment for things like this where it might not be 100% clear. 

Do you use any numpy functions? If not, consider using base python arrays - they tend to be a little more readable. Alternatively, consider using numpy's functions! I'm certain you could do lines 47-60 in a single (fairly unreadable) line or just a couple of readable lines. For example:

Function and variable names should be more descriptive (and PEP-8 compliant)
- maxgwidth? rend? lstart? what are these things? It may make sense to you but as someone glancing over your code it's much harder to read these.

Try to avoid excessive indentation - this will improve readability
instead of 
```
if x:
  y
else:
  if w:
    z
  else:
    if ...
```
try 
```
if x:
  y
elif w:
  z
else:
  ...
```
On that note, elif is more efficient than nested ifs (and much easier to read).

Please include spaces between =, commas, etc. to increase readability. 

What are lines 155-159 doing? I'm not sure what that index is exactly.

On line 154, you should consider using <= 2 instead of < 3 - it's functionally equivalent, but semantically closer to what you're doing. This will help increase readability (sorry for saying it again)

Your 2D array notation might work but doesn't look very pythonic - core python doesn't have 2D arrays like `a[2,4]` (but numpy does). As such, 2D arrays are normally nested lists, i.e. `a[2][4]`. You may not see the advantage of this straight away, but it would allow you to do e.g. on lines 120-124:
```
lines[y] = [
  lstart,
  lend,
  gwidth,
  rstart,
  rend
]
```
Which is much more readable - readability should be maximised for python code. 

From an implementation (rather than code style) POV - is loading and saving files cheating in terms of training time? The intensive bit is the training, and that appears to be done beforehand. Additionally, .npy files are weird - consider saving to a csv instead if you must continue doing this (then at least it's human-readable) 

We've already mentioned that we'll need to evaluate against the training data, so I won't mention that again in detail. 

These are the main things I've seen - there might be other things, but these few tips should massively help with future code collaboration (I hope!)
