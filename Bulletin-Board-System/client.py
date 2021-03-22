import threading
from itertools import count

from amazon_s3 import S3

class Client(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn
        self.s3 = S3()

    def run(self):
        # welcome user
        welcome = self.recieveSingle()
        print(welcome)
        welcome = self.recieveSingle()
        print(welcome)
        welcome = self.recieveSingle()
        print(welcome)

        # fp = open('./test_file/delbuck.txt', 'r')  # 'add.txt', 'mail.txt', 'delbuck.txt'
        # cmd_L = fp.readlines()
        # fp.close()

        # for i, cmd in enumerate(cmd_L):
        #     self.sendCmd(cmd.strip('\n'))
        #     if cmd == 'exit':
        #         break

        while True:
            self.recieveSingle()
            print('\n% ', end='')
            cmd = input()
            self.sendCmd(cmd)
            if (cmd == 'exit'):
                break

    def sendResponse(self, response):
        self.conn.send(str(response).encode('utf-8'))

    def receiveResponse(self):
        while True:
            data = self.conn.recv(1024)
            if data:
                data = data.decode().strip()
                return data

    def recieveSingle(self):
        data = self.receiveResponse()
        self.sendResponse('OK')
        return data

    def receiveData(self):
        sentence = self.recieveSingle()
        permit = self.recieveSingle()

        metadata = None
        if permit == 'YES':
            metadata = self.recieveSingle()

        return sentence, permit, metadata

    def sendCmd(self, cmd):
        self.sendResponse(cmd)

        if (cmd == 'exit'):
            return
        elif (cmd.startswith('register')):
            self.register()
        elif (cmd.startswith('create-post')):
            self.createPost()
        elif (cmd.startswith('list-board') or cmd.startswith('list-post') or cmd == 'list-mail'):
            self.listTrecv()
        elif (cmd.startswith('read')):
            self.readPost()
        elif (cmd.startswith('delete-post')):
            self.delPost()
        elif (cmd.startswith('update-post')):
            self.updatePost()
        elif (cmd.startswith('comment')):
            self.comment()
        elif (cmd.startswith('listbuck')):
            self.s3.list_all()
        elif (cmd.startswith('delbuck')):
            self.s3.delete_all()
            self.s3.create_bucket('pinyugly')
        elif (cmd.startswith('mail-to')):
            self.sendMail()
        elif (cmd.startswith('retr-mail')):
            self.retrMail()
        elif (cmd.startswith('delete-mail')):
            self.deleteMail()
        else:
            sentence, _, _ = self.receiveData()
            print(sentence)

    def register(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            self.s3.create_bucket(bucketName)
            self.s3.create_object(bucketName, 'mail/')

    def listTrecv(self):
        permit = self.recieveSingle()
        if (permit == 'YES'):
            while True:
                sentence = self.recieveSingle()
                if (sentence == 'END'):
                    break
                elif (sentence == 'NEWLINE'):
                    print('\n', end='')
                else:
                    print(('%-20s')%sentence, end='')
        else:
            sentence, _, _ = self.receiveData()
            print(sentence)

    def createPost(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            postId = self.recieveSingle()
            content = self.recieveSingle()
            self.s3.create_object(bucketName, postId + '/')
            self.s3.create_object(bucketName, postId + '/content', content)

    def readPost(self):
        permit = self.recieveSingle()
        if (permit == 'YES'):
            # info
            while True:
                sentence = self.recieveSingle()
                if(sentence == 'END'):
                    break
                else:
                    print(sentence)

            # content
            bucketName = self.recieveSingle()
            postId = self.recieveSingle()
            content = self.s3.get_object(bucketName, postId + '/content')
            content_split = content.split('<br>')
            for row in content_split:
                print(row.strip())

            # comment
            sentence = self.recieveSingle()
            print(sentence)
            while True:
                com_permit = self.recieveSingle()
                if (com_permit == 'NO'):
                    break

                comId = self.recieveSingle()
                comAuthor = self.recieveSingle()
                content = self.s3.get_object(bucketName, '{}/{}'.format(postId, comId))
                print('{}: {}'.format(comAuthor, content))
        else:
            sentence, _, _ = self.receiveData()
            print(sentence)

    def updatePost(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            objName = self.recieveSingle()
            content = self.recieveSingle()
            self.s3.create_object(bucketName, objName, content)

    def delPost(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            fileName = self.recieveSingle()
            self.s3.delete_file(bucketName, fileName)

    def comment(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            objName = self.recieveSingle()
            content = self.recieveSingle()
            self.s3.create_object(bucketName, objName, content)

    def sendMail(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            objName = self.recieveSingle()
            content = self.recieveSingle()
            self.s3.create_object(bucketName, objName, content)

    def retrMail(self):
        permit = self.recieveSingle()
        if (permit == 'YES'):
            # info
            while True:
                sentence = self.recieveSingle()
                if (sentence == 'END'):
                    break
                else:
                    print(sentence)

            # content
            bucketName = self.recieveSingle()
            mailId = self.recieveSingle()
            content = self.s3.get_object(bucketName, mailId)
            content_split = content.split('<br>')
            for row in content_split:
                print(row.strip())

        else:
            sentence, _, _ = self.receiveData()
            print(sentence)

    def deleteMail(self):
        sentence, permit, bucketName = self.receiveData()
        print(sentence)
        if (permit == 'YES'):
            objName = self.recieveSingle()
            self.s3.delete_object(bucketName, objName)

