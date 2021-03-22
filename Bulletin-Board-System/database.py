import sqlite3

import globalVars as GV


class bbsDatabase():
	def __init__(self):
		# open database
		self.conn = sqlite3.connect('%s.db'%GV.BBSname)
		
		# SQL
		self.cursor = self.conn.cursor()

	def createTable(self):
		# create user table
		self.cursor.execute('''CREATE TABLE IF NOT EXISTS %s
							(ID         INTEGER PRIMARY KEY   AUTOINCREMENT,
							 USERNAME   TEXT 				  NOT NULL,
							 EMAIL      TEXT 				  NOT NULL,
							 PASSWORD   TEXT     			  NOT NULL);'''%GV.userTname)
		self.conn.commit()

		# create board table
		self.cursor.execute('''CREATE TABLE IF NOT EXISTS %s
							(ID        INTEGER PRIMARY KEY  AUTOINCREMENT,
							 NAME      TEXT    				NOT NULL,
							 MODERATOR TEXT     		    NOT NULL);'''%GV.boardTname)
		self.conn.commit()

		# create post table
		self.cursor.execute('''CREATE TABLE IF NOT EXISTS %s
							(ID        INTEGER PRIMARY KEY  AUTOINCREMENT,
							 TITLE     TEXT    				NOT NULL,
							 AUTHOR    TEXT    				NOT NULL,
							 DATE      TEXT    				NOT NULL,
							 BOARD     TEXT    				NOT NULL);'''%GV.postTname)

		# CONTENT   TEXT    				NOT NULL,

		self.conn.commit()

		# create comment table
		self.cursor.execute('''CREATE TABLE IF NOT EXISTS %s
							(ID        INTEGER PRIMARY KEY  AUTOINCREMENT,
							 POST_ID   INT    				NOT NULL,
							 AUTHOR    TEXT    				NOT NULL);'''%GV.comTname)

		# create mail table
		self.cursor.execute('''CREATE TABLE IF NOT EXISTS %s
							(ID        INTEGER PRIMARY KEY  AUTOINCREMENT,
							 RECEIVER  TEXT    				NOT NULL,
							 SUBJECT   TEXT    				NOT NULL,
							 SENDER    TEXT    				NOT NULL,
							 DATE      TEXT    				NOT NULL);''' % GV.mailTname)

		self.conn.commit()

	def insertUser(self, usrName, email, pw): # register
		self.cursor.execute("INSERT into {} (USERNAME,EMAIL,PASSWORD) \
				 			 VALUES ('{}','{}','{}')".format(GV.userTname, usrName, email, pw))
		self.conn.commit()
		return str(self.cursor.lastrowid)

	def insertBoard(self, name, moderator):
		self.cursor.execute("INSERT into {} (NAME,MODERATOR) \
				 			 VALUES ('{}','{}')".format(GV.boardTname, name, moderator))
		self.conn.commit()	

	def insertPost(self, title, author, date, board):
		self.cursor.execute("INSERT into {} (TITLE,AUTHOR,DATE,BOARD) \
				 			 VALUES ('{}','{}','{}','{}')".format(GV.postTname, title, author, date, board))
		self.conn.commit()
		return str(self.cursor.lastrowid)

	def insertComment(self, postID, author):
		self.cursor.execute("INSERT into {} (POST_ID,AUTHOR) \
				 			 VALUES ('{}','{}')".format(GV.comTname, postID, author))
		self.conn.commit()
		return str(self.cursor.lastrowid)

	def insertMail(self, receiver, subject, sender, date):
		self.cursor.execute("INSERT into {} (RECEIVER,SUBJECT,SENDER,DATE) \
				 			 VALUES ('{}','{}','{}','{}')".format(GV.mailTname, receiver, subject, sender, date))
		self.conn.commit()
		return str(self.cursor.lastrowid)

	def findAllData(self, tableName, keyWord=None):
		self.cursor.execute("SELECT * from {} {}".format(tableName, keyWord))
		return self.cursor.fetchall() # list

	def findTitle(self, tableName):
		# pragma table_info(table_name)
		self.cursor.execute("pragma table_info({})".format(tableName))
		return self.cursor.fetchall()

	def findData(self, name, table):
		if(table==GV.userTname):
			title = 'USERNAME'
		elif(table==GV.boardTname):
			title = 'NAME'
		elif(table==GV.postTname):
			title = 'ID'
		elif(table==GV.comTname):
			title = 'POST_ID'
		elif(table==GV.mailTname):
			title = 'RECEIVER'

		self.cursor.execute("SELECT * from {} WHERE {} = '{}'".format(table, title, name))

		return self.cursor.fetchall()


	def updatePost(self, colName, new, postID):
		self.cursor.execute("UPDATE {} SET {} = '{}' \
							 WHERE ID = '{}'".format(GV.postTname, colName, new, postID))

		self.conn.commit()

	def delPost(self, postId):
		# post
		self.cursor.execute("DELETE FROM {} \
							 WHERE ID = '{}'".format(GV.postTname, postId))

		# comment
		self.cursor.execute("DELETE FROM {} \
							 WHERE POST_ID = '{}'".format(GV.comTname, postId))

		self.conn.commit()

	def delMail(self, mailId):
		self.cursor.execute("DELETE FROM {} \
							 WHERE ID = '{}'".format(GV.mailTname, mailId))

		self.conn.commit()

	def close(self):
		self.conn.close()
