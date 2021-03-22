import threading

# global variables
def initialize(): 
	global clients
	global CONNlock
	global DBlock
	global BBSname, userTname, boardTname, postTname, comTname, mailTname
	global cPostPat, lPostPat1, lPostPat2, uPostPat1, uPostPat2, mail2Pat
	global bucketPre

	clients = [] 
	CONNlock = threading.Lock()
	DBlock = threading.Lock()
	BBSname, userTname, boardTname, postTname, comTname, mailTname = 'BBS', 'user', 'board', 'post', 'comment', 'mail'
	cPostPat = r'create-post (.*) --title (.*) --content (.*)'
	lPostPat1 = r'list-post (.*) ##(.*)'
	lPostPat2 = r'list-post (.*)'
	uPostPat1 = r'update-post ([1-9][0-9]*) --title (.*)'
	uPostPat2 = r'update-post ([1-9][0-9]*) --content (.*)'
	mail2Pat = r'mail-to (.*) --subject (.*) --content (.*)'
	bucketPre = 'pinyugly'