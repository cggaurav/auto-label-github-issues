const axios = require('axios')
const Promise = require('bluebird')
const fs = require('fs')

const FILENAME = './data.csv'
const ISSUES_PER_PAGE = 100
const WAIT_TIME_BEFORE_REQUESTS = 50

const LABELS = [
	"bug",
	"question",
	"enhancement",
	"feature",
	"help wanted",
	"doc" // documentation
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
	"ansible/ansible",

	// Maybe
	// "FortAwesome/Font-Awesome",
	// "npm/npm",
	// "golang/go"
]

// DOCS
// https://developer.github.com/v3/
// https://api.github.com/repos/kubernetes/kubernetes/issues?state=closed&page=2&per_page=50

// TODO: Rate limiting
// TODO: Pagination

function writeIssueToFile(issue, labels) {
	// FORMAT: https://api.github.com/repos/kubernetes/kubernetes/issues?state=closed&page=1&per_page=1
	console.log(`Writing issue ${issue.url} with labels ${JSON.stringify(labels)}`)

	fs.appendFileSync(FILENAME, [[issue.url, issue.id, `${(issue.title || '').replace(/,/g, ' ').replace(/\r|\n/g, ' ')}`, `${(issue.body || '').replace(/,/g, ' ').replace(/\r|\n/g, ' ')}`, Array.from(new Set(labels)).join('|')].join(','), '\n'].join(''))
}


function makeGithubIssueRequest(repository, page = 1) {

	return new Promise((resolve, reject) => {
		let ENDPOINT = `https://api.github.com/repos/${repository}/issues?state=closed&page=${page}&per_page=${ISSUES_PER_PAGE}`

		console.log(`Making a request for ${repository} and page ${page}`)

		console.log(process.env.GITHUB_TOKEN)

		return axios.get(ENDPOINT, {
			headers: {
				'Authorization': `token ${process.env.GITHUB_TOKEN}`
			}
		})
		.then((body) => {
			// FORMAT: https://api.github.com/repos/kubernetes/kubernetes/issues?state=closed&page=1&per_page=1
			const issues = body.data

			if (issues.length === 0) {
				return resolve()
			}

			issues.forEach((issue, index) => {

				let _labels = []

				issue.labels.forEach((issue_label) => {
					LABELS.forEach((label) => {
						if (issue_label.name.toLowerCase().includes(label)) {
							_labels.push(label)
						}
					})
				})
				
				if (_labels.length > 0) {
					writeIssueToFile(issue, _labels)
				}
			})

			setTimeout(() => {
				return makeGithubIssueRequest(repository, page + 1)
			}, WAIT_TIME_BEFORE_REQUESTS)

		})
		.catch((err) => {
			// console.error(err)
			return reject(err)
		})	
	})
}

Promise.each(REPOSITORES, (repository) => {
	return new Promise((resolve, reject) => {

		return makeGithubIssueRequest(repository)
		
	})
})