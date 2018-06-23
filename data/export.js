const axios = require('axios')
const Promise = require('bluebird')
const fs = require('fs')

const FILENAME = './data.csv'

const LABELS = [
	"bug",
	"question",
	"enhancement",
	"feature",
	"help wanted"
	// "security",
	// "test"
]

const REPOSITORES = [
	"kubernetes/kubernetes",
	"openshift/origin",
	"cms-sw/cmssw",
	"Microsoft/vscode",
	"rust-lang/rust",
	"dotnet/corefx",
	"tgstation/tgstation",
	"nodejs/node",
	"servo/servo",
	"ansible/ansible"
]

// DOCS
// https://developer.github.com/v3/
// https://api.github.com/repos/kubernetes/kubernetes/issues?state=closed&page=2&per_page=50

// TODO: Rate limiting
// TODO: Pagination

Promise.each(REPOSITORES, (repository) => {
	return new Promise((resolve, reject) => {
		let ENDPOINT = `https://api.github.com/repos/${repository}/issues?state=closed&page=1&per_page=10`
		return axios.get(ENDPOINT)
		  	.then((body) => {
		  		// FORMAT: https://api.github.com/repos/kubernetes/kubernetes/issues?state=closed&page=1&per_page=1
		  		const issues = body.data

		  		issues.forEach((issue) => {

		  			let _labels = []

		  			issue.labels.forEach((issue_label) => {
		  				LABELS.forEach((label) => {
		  					if (issue_label.name.toLowerCase().includes(label)) {
		  						_labels.push(label)
		  					}
		  				})
		  			})
		  			
		  			if (_labels.length > 0) {
		  				fs.appendFileSync(FILENAME, [[issue.url, issue.id, `${issue.body.replace(/,g/, '')}`, _labels].join(','), '\n'].join(''))
		  				return resolve(_labels)
		  			} else {
		  				return resolve(null)
		  			}
		  		})
		  	})
		  	.catch((err) => {
		  		console.error(err)
		  	})
	})
})