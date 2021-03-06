# auto-label-github issues (beta)

This project exposes an LSTM model that auto labels Github issues. The goal is to have the full pipeline in continous deployment and rapidly improve the model accuracy. This is an independant project to clear the cloud on how to actually build an ML model and understand what it takes for taking it from 0 - 1 with continues integration.

## Training set

Following [Octoverse](https://octoverse.github.com/), we looked at the "Ten most discussed repositores" and trained on the closed
issues.


## Supported Labels

After looking at Github's [default](https://help.github.com/articles/about-labels/) labels and what's being used in the projets above we as a first pass are training. See [Labels](app/server/data/data.labels.csv)

```
bug
question
enhancement
feature
help wanted
doc
```

## Training Strategy

We are picking the `title` and mapping the existing labels in the issues above to our baseline version. We are using [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

We are using `pytorch v0.3.1`

## Demo

Here you [go](https://cggaurav.net/auto-label-github-issues/) and the backend API sits [here](https://algi.herokuapp.com/)

## Contribute

TODO

* Train more labels
* Output multi labels
