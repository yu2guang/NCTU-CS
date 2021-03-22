import threading
import time
import re
from amazon_s3 import S3
from database import bbsDatabase
import globalVars as GV


class Server(threading.Thread):
	def __init__(self, conn, addr):
		threading.Thread.__init__(self)
		self.conn = conn
		self.addr = addr

	def run(self):
		# client connected
		GV.CONNlock.acquire()
		GV.clients.append(self)
		GV.CONNlock.release()
		print('\nNew connection: {}'.format(self.addr))

		# connect to DB		
		self.DB = bbsDatabase()
		self.client = S3()

		# welcome user
		self.sendSingle('********************************\n')
		self.sendSingle('** Welcome to the BBS server. **\n')
		self.sendSingle('********************************\n')

		self.userName = ''
		self.userBucketName = ''
		self.logStatus = False

		while True:
			self.sendSingle('%')
			cmd = self.receiveResponse()
			print(cmd)
			self.processCmd(cmd)
			if(cmd=='exit'):
				break

	def sendResponse(self, response):
		self.conn.send(str(response).encode('utf-8'))		

	def receiveResponse(self):
		while True:
			cmd = self.conn.recv(1024)
			if cmd:
				cmd = cmd.decode().strip()
				return cmd

	def sendSingle(self, data):
		self.sendResponse(data)
		self.receiveResponse()

	def sendData(self, sentence, permit, metadata=None):
		self.sendSingle(sentence)
		self.sendSingle(permit)
		if permit == 'YES':
			self.sendSingle(metadata)

	def patMatch(self, pattern, inputStr):
		return re.match(pattern, inputStr)

	def listT(self, tableName, keyWord=None):
		title = self.DB.findTitle(tableName)
		data = self.DB.findAllData(tableName, keyWord)

		if(title and data):
			self.sendSingle('YES')

			for i, row in enumerate(title):
				if(i==0):
					self.sendSingle(('%-5s')%row[1])
				elif(tableName==GV.mailTname and i==1):
					continue
				elif(tableName==GV.postTname and i==4):
					break
				else:
					self.sendSingle(('%-20s')%row[1])
			self.sendSingle('NEWLINE')

			for mailId, row in enumerate(data, start=1):
				for i, item in enumerate(row):
					if(i==0):
						if(tableName==GV.mailTname):
							self.sendSingle(('%-5s')%mailId)
						else:
							self.sendSingle(('%-5s')%item)
					elif(tableName==GV.mailTname and i==1):
						continue
					elif((tableName==GV.postTname and i==3) or (tableName==GV.mailTname and i==4)):
						_, month, day = item.split('-')
						item_con = month+'/'+day
						self.sendSingle(('%-20s')%item_con)
						break
					else:
						self.sendSingle(('%-20s')%item)
				self.sendSingle('NEWLINE')

			self.sendSingle('END')
		else:
			self.sendSingle('NO')
			sentence = 'There\'s no data.\n'
			self.sendData(sentence, 'NO')

	def processCmd(self, cmd):
		if(cmd.startswith('register')):
			self.register(cmd)
		elif(cmd.startswith('login')):
			self.login(cmd)
		elif(cmd=='logout'):
			self.logout()
		elif(cmd=='whoami'):
			self.whoami()
		elif(cmd=='exit'):
			self.exit()
		elif(cmd.startswith('create-board')):
			self.createBoard(cmd)
		elif(cmd.startswith('create-post')):
			self.createPost(cmd)
		elif(cmd.startswith('list-board')):
			self.listBoard(cmd)
		elif(cmd.startswith('list-post')):
			self.listPost(cmd)
		elif(cmd.startswith('read')):
			self.readPost(cmd)
		elif(cmd.startswith('delete-post')):
			self.delPost(cmd)
		elif(cmd.startswith('update-post')):
			self.updatePost(cmd)
		elif(cmd.startswith('comment')):
			self.comment(cmd)
		elif (cmd.startswith('mail-to')):
			self.sendMail(cmd)
		elif (cmd=='list-mail'):
			self.listMail()
		elif (cmd.startswith('retr-mail')):
			self.retrMail(cmd)
		elif (cmd.startswith('delete-mail')):
			self.deleteMail(cmd)
		else:
			self.sendResponse('{}: command not found.\n'.format(cmd))		

	def register(self, cmd):
		cmd_split = cmd.strip().split()
		if(len(cmd_split)!=4):
			sentence = 'Usage: register <username> <email> <password>\n'
			self.sendData(sentence, 'NO')
		else:
			_, name, mail, pw = cmd_split
			GV.DBlock.acquire()
			used = self.DB.findData(name, GV.userTname)
			if(used):
				sentence = 'Username is already used.\n'
				self.sendData(sentence, 'NO')
			else:
				userId = self.DB.insertUser(name, mail, pw)
				sentence = 'Register successfully.\n'
				bucketName = '{}-{}-{}'.format(GV.bucketPre, name.lower(), userId)
				self.sendData(sentence, 'YES', bucketName)
			GV.DBlock.release()

	def login(self, cmd):
		cmd_split = cmd.strip().split()
		if(len(cmd_split)!=3):
			sentence = 'Usage: login <username> <password>\n'
			self.sendData(sentence, 'NO')
		else:
			_, name, pw = cmd_split
			GV.DBlock.acquire()
			data = self.DB.findData(cmd_split[1], GV.userTname)
			pwDB = None
			if(data): 
				Id, pwDB = data[0][0], data[0][3]
			
			if(self.logStatus):
				sentence = 'Please logout first.\n'
				self.sendData(sentence, 'NO')
			elif(data and pw==pwDB):
				self.userName = name
				self.userBucketName = '{}-{}-{}'.format(GV.bucketPre, name.lower(), Id)
				self.logStatus = True
				sentence = 'Welcome, {}.\n'.format(name)
				self.sendData(sentence, 'NO')
			else:
				sentence = 'Login failed.\n'
				self.sendData(sentence, 'NO')
			GV.DBlock.release()

	def logout(self):
		if(self.logStatus):
			sentence = 'Bye, {}.\n'.format(self.userName)
			self.sendData(sentence, 'NO')
			self.userName = ''
			self.userBucketName = ''
			self.logStatus = False
		else:
			sentence = 'Please login first.\n'
			self.sendData(sentence, 'NO')

	def whoami(self):
		if(self.logStatus):
			sentence = '{}.\n'.format(self.userName)
			self.sendData(sentence, 'NO')
		else:
			sentence = 'Please login first.\n'
			self.sendData(sentence, 'NO')

	def exit(self):
		GV.CONNlock.acquire()
		GV.clients.remove(self)
		GV.CONNlock.release()
		
		self.DB.close()
		self.conn.close()

		print('\nDisconnection: {}'.format(self.addr))	

	def createBoard(self, cmd):
		cmd_split = cmd.strip().split(' ', 1)
		if(len(cmd_split)!=2):
			sentence = 'Usage: create-board <name>\n'
			self.sendData(sentence, 'NO')
		else:
			if(self.logStatus):
				_, name = cmd_split
				GV.DBlock.acquire()
				used = self.DB.findData(name, GV.boardTname)
				if(used):
					sentence = 'Board already exists.\n'
					self.sendData(sentence, 'NO')
				else:
					self.DB.insertBoard(name, self.userName)
					sentence = 'Create board successfully.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')

	def createPost(self, cmd):
		matchCmd = self.patMatch(GV.cPostPat, cmd)

		if(matchCmd):
			if(self.logStatus):
				_, rest = cmd.strip().split(' ', 1)
				boardName, rest = rest.strip().split('--title', 1)
				boardName = boardName.strip()
				title, content = rest.strip().split('--content', 1)
				title = title.strip()
				content = content.strip()

				GV.DBlock.acquire()
				existed = self.DB.findData(boardName, GV.boardTname)
				if(existed):
					date = time.strftime("%Y-%m-%d", time.localtime())
					postId = self.DB.insertPost(title, self.userName, date, boardName)
					sentence = 'Create post successfully.\n'
					self.sendData(sentence, 'YES', self.userBucketName)
					self.sendSingle(postId)
					self.sendSingle(content)
				else:
					sentence = 'Board does not exist.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')
		else:
			sentence = 'Usage: create-post <board-name> --title <title> --content <content>\n'
			self.sendData(sentence, 'NO')

	def listBoard(self, cmd):
		cmd_split = cmd.strip().split(' ', 1)
		if(len(cmd_split)==1):
			GV.DBlock.acquire()
			self.listT(GV.boardTname)
			GV.DBlock.release()
		elif(len(cmd_split)==2 and cmd_split[1].startswith('##')):
			keyWord = cmd_split[1].lstrip('##')
			GV.DBlock.acquire()
			self.listT(GV.boardTname, 'WHERE NAME LIKE \'%{}%\''.format(keyWord))
			GV.DBlock.release()
		else:
			self.sendSingle('NO')
			sentence = 'Usage: list-board ##<key>\n'
			self.sendData(sentence, 'NO')

	def listPost(self, cmd):
		# list-post <board-name> ##<key>
		matchCmd1 = self.patMatch(GV.lPostPat1, cmd)
		# list-post <board-name>
		matchCmd2 = self.patMatch(GV.lPostPat2, cmd)

		if(matchCmd1 or matchCmd2):
			_, rest = cmd.strip().split(' ', 1)
			boardName = rest.strip().split('##', 1)[0]
			boardName = boardName.strip()

			GV.DBlock.acquire()
			existed = self.DB.findData(boardName, GV.boardTname)
			if(existed):
				if(matchCmd1): # key
					keyWord = rest.strip().split('##', 1)[1].strip()
					self.listT(GV.postTname, 'WHERE BOARD LIKE \'{}\' AND TITLE LIKE \'%{}%\''.format(boardName, keyWord))
				else:
					self.listT(GV.postTname, 'WHERE BOARD LIKE \'{}\''.format(boardName))
			else:
				self.sendSingle('NO')
				sentence = 'Board does not exist.\n'
				self.sendData(sentence, 'NO')
			GV.DBlock.release()
		else:
			self.sendSingle('NO')
			sentence = 'Usage: list-post <board-name> ##<key>\n'
			self.sendData(sentence, 'NO')
	
	def readPost(self, cmd):
		cmd_split = cmd.strip().split(' ', 1)
		if(len(cmd_split)!=2):
			self.sendSingle('NO')
			sentence = 'Usage: read <post-id>\n'
			self.sendData(sentence, 'NO')
		else:
			_, postId = cmd_split	
			GV.DBlock.acquire()
			data = self.DB.findData(postId, GV.postTname)
			
			if(data):
				self.sendSingle('YES')

				_, title, author, date, _ = data[0]
				self.sendSingle('Author: {}\n'.format(author))
				self.sendSingle('Title : {}\n'.format(title))
				self.sendSingle('Date  : {}\n'.format(date))
				
				# content
				self.sendSingle('\n--\n\n')
				self.sendSingle('END')
				userData = self.DB.findData(author, GV.userTname)
				userId = userData[0][0]
				bucketName = '{}-{}-{}'.format(GV.bucketPre, author.lower(), userId)
				self.sendSingle(bucketName)
				self.sendSingle(postId)
				
				# comment
				self.sendSingle('\n--\n\n')
				comments = self.DB.findData(postId, GV.comTname)
				if(comments):
					for row in comments:
						self.sendSingle('YES')
						comId, _, comAuthor = row
						self.sendSingle(comId)
						self.sendSingle(comAuthor)
					self.sendSingle('NO')
				else:
					self.sendSingle('NO')

						
				self.sendResponse('\n')
			else:
				self.sendSingle('NO')
				sentence = 'Post does not exist.\n'
				self.sendData(sentence, 'NO')
			GV.DBlock.release()	

	def delPost(self, cmd):
		cmd_split = cmd.strip().split(' ')
		if(len(cmd_split)!=2):
			sentence = 'Usage: delete-post <post-id>\n'
			self.sendData(sentence, 'NO')
		else:
			if(self.logStatus):
				_, postID = cmd_split
				postID = postID.strip()

				GV.DBlock.acquire()
				data = self.DB.findData(postID, GV.postTname)
				if(data):
					_, _, author, _, _ = data[0]
					if(self.userName==author):
						self.DB.delPost(postID)
						sentence = 'Delete successfully.\n'
						self.sendData(sentence, 'YES', self.userBucketName)
						self.sendSingle(postID + '/')
					else:
						sentence = 'Not the post owner.\n'
						self.sendData(sentence, 'NO')
				else:
					sentence = 'Post does not exist.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')

	def updatePost(self, cmd):
		# update-post <post-id> --title <new>
		matchCmd1 = self.patMatch(GV.uPostPat1, cmd)
		# update-post <post-id> --content <new>
		matchCmd2 = self.patMatch(GV.uPostPat2, cmd)

		if(matchCmd1 or matchCmd2):
			if(self.logStatus):
				_, postId, restCmd = cmd.strip().split(' ', 2)
				postId = postId.strip()

				GV.DBlock.acquire()
				data = self.DB.findData(postId, GV.postTname)
				if(data):
					_, _, author, _, _ = data[0]
					if(self.userName==author):
						TC, new = restCmd.strip().split(' ', 1)
						TC = TC.strip().lstrip('--')
						new = new.strip()

						sentence = 'Update {} successfully.\n'.format(TC)

						if(matchCmd1):
							self.DB.updatePost(TC, new, postId)
							self.sendData(sentence, 'NO')
						else:
							self.sendData(sentence, 'YES', self.userBucketName)
							self.sendSingle(postId + '/content')
							self.sendSingle(new)

					else:
						sentence = 'Not the post owner.\n'
						self.sendData(sentence, 'NO')
				else:
					sentence = 'Post does not exist.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')
		else:
			sentence = 'Usage: update-post <post-id> --title/content <new>\n'
			self.sendData(sentence, 'NO')

	def comment(self, cmd):
		cmd_split = cmd.strip().split(' ', 2)
		if(len(cmd_split)!=3):
			sentence = 'Usage: comment <post-id> <comment>\n'
			self.sendData(sentence, 'NO')
		else:
			if(self.logStatus):
				_, postId, content = cmd_split
				postId = postId.strip()
				content = content.strip()

				GV.DBlock.acquire()
				postData = self.DB.findData(postId, GV.postTname)
				if(postData):
					comId = self.DB.insertComment(postId, self.userName)
					author = postData[0][2]
					userData = self.DB.findData(author, GV.userTname)
					userId = userData[0][0]
					bucketName = '{}-{}-{}'.format(GV.bucketPre, author.lower(), userId)
					sentence = 'Comment successfully.\n'
					self.sendData(sentence, 'YES', bucketName)
					self.sendSingle('{}/{}'.format(postId, comId))
					self.sendSingle(content)
				else:
					sentence = 'Post does not exist.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')

	def sendMail(self, cmd):
		matchCmd = self.patMatch(GV.mail2Pat, cmd)

		if(matchCmd):
			if(self.logStatus):
				_, rest = cmd.strip().split(' ', 1)
				receiver, rest = rest.strip().split('--subject', 1)
				receiver = receiver.strip()
				subject, content = rest.strip().split('--content', 1)
				subject = subject.strip()
				content = content.strip()
				sender = self.userName

				GV.DBlock.acquire()
				receiverData = self.DB.findData(receiver, GV.userTname)
				if(receiverData):
					date = time.strftime("%Y-%m-%d", time.localtime())
					bucketName = '{}-{}-{}'.format(GV.bucketPre, receiver.lower(), receiverData[0][0])
					mailId = self.DB.insertMail(receiver, subject, sender, date)
					sentence = 'Send successfully.\n'
					self.sendData(sentence, 'YES', bucketName)
					self.sendSingle('mail/' + str(mailId))
					self.sendSingle(content)
				else:
					sentence = '{} does not exist.\n'.format(receiver)
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')
		else:
			sentence = 'Usage: mail-to <username> --subject <subject> --content <content>\n'
			self.sendData(sentence, 'NO')

	def listMail(self):
		if (self.logStatus):
			GV.DBlock.acquire()
			self.listT(GV.mailTname, 'WHERE RECEIVER = \'{}\''.format(self.userName))
			GV.DBlock.release()
		else:
			self.sendSingle('NO')
			sentence = 'Please login first.\n'
			self.sendData(sentence, 'NO')

	def retrMail(self, cmd):
		cmd_split = cmd.strip().split(' ', 1)
		if (len(cmd_split) != 2):
			self.sendSingle('NO')
			sentence = 'Usage: retr-mail <mail#>\n'
			self.sendData(sentence, 'NO')
		else:
			if (self.logStatus):
				_, mailId = cmd_split
				GV.DBlock.acquire()
				userMailData = self.DB.findData(self.userName, GV.mailTname)

				if (userMailData and len(userMailData) >= int(mailId)):
					self.sendSingle('YES')

					mailId_real, _, subject, sender, date = userMailData[int(mailId)-1]
					self.sendSingle('Subject: {}\n'.format(subject))
					self.sendSingle('From   : {}\n'.format(sender))
					self.sendSingle('Date   : {}\n'.format(date))

					# content
					self.sendSingle('\n--\n\n')
					self.sendSingle('END')
					self.sendSingle(self.userBucketName)
					self.sendSingle('mail/' + str(mailId_real))

				else:
					self.sendSingle('NO')
					sentence = 'No such mail.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				self.sendSingle('NO')
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')


	def deleteMail(self, cmd):
		cmd_split = cmd.strip().split(' ', 1)
		if (len(cmd_split) != 2):
			sentence = 'Usage: delete-mail <mail#>\n'
			self.sendData(sentence, 'NO')
		else:
			if (self.logStatus):
				_, mailId = cmd_split
				GV.DBlock.acquire()
				userMailData = self.DB.findData(self.userName, GV.mailTname)

				if (userMailData and len(userMailData) >= int(mailId)):
					mailId_real = userMailData[int(mailId)-1][0]
					self.DB.delMail(mailId_real)
					sentence = 'Mail deleted.\n'
					self.sendData(sentence, 'YES', self.userBucketName)
					self.sendSingle('mail/' + str(mailId_real))
				else:
					sentence = 'No such mail.\n'
					self.sendData(sentence, 'NO')
				GV.DBlock.release()
			else:
				sentence = 'Please login first.\n'
				self.sendData(sentence, 'NO')
