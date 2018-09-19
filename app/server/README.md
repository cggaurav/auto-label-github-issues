##

API backend from which the model is served. Experimented with ones below for "easy + cheap" deploys.

### Heroku Python Runtime

Stopped after an issue with slug size > 500mb

`git push heroku `git subtree split --prefix app/server master`:master --force`
`git subtree push --prefix app/server master`

### Floydhub

Using floyd to deploy the model out. Floyd is great as the Heroku for ML though their serve is in beta and not 24x7.

### K8

Going K8 with Dockerfile at last. There is so much work needed in taking a model to production in terms of deployments, SSL Proxy, CORS, etc all.

### Heroku Containers

Seems to work since slug size is not an issue and images are pushed to their own registry! [Dockerfile](https://github.com/cggaurav/auto-label-github-issues/blob/master/app/Dockerfile) using Jessie

