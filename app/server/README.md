##

API backend from which the model is served

### Heroku

Stopped after an issue with slug size > 500mb

`git push heroku `git subtree split --prefix app/server master`:master --force`
`git subtree push --prefix app/server master`

### Floydhub

Using floyd to deploy the model out. Floyd is great as the Heroku for ML though their serve is in beta and not 24x7.

### K8

Going K8 with Dockerfile at last. 
