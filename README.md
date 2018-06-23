# auto-label-github issues

## Training set

Following [Octoverse](https://octoverse.github.com/), we looked at the "Ten most discussed repositores" and trained on the closed
issues.


## Supported Labels

After looking at Github's [default](https://help.github.com/articles/about-labels/) labels and what's being used in the projets above we as a first pass are training.

```
bug
question
enhancement
feature
help wanted
```

## Training Strategy

We are picking the `title`, the first `thread` and mapping the existing labels in the issues above to our initial version. Our baseline implementation is an [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)



## Demo

Here you [go]()