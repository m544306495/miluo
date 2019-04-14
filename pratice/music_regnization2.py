import os

import MySQLdb

import  music_regnization1


class memory():
    def __init__(self, host, port, user, passwd, db):
        '''
        初始化的方法，主要是存储连接数据库的参数
        :param host:
        :param port:
        :param user:
        :param passwd:
        :param db:
        '''
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db

    def addsong(self, path):
        '''
        添加歌曲方法，将歌曲名和歌曲特征指纹存到数据库
        :param path: 歌曲路径
        :return:
        '''
        if type(path) != str:
            raise TypeError
            print('path need string')
        basename = os.path.basename(path)
        try:
            conn = MySQLdb.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                                   charset='utf8')
        except:
            print ('DataBase error1')
            return None
        cur = conn.cursor()
        namecount = cur.execute("select * from test.music WHERE song_name = '%s'" % basename)
        if namecount > 0:
            print ('the song has been record!')
            return None
        v = music_regnization1.voice()
        v.loaddata(path)
        v.fft()
        cur.execute("insert into test.music VALUES('%s','%s')" % (basename, v.high_point.__str__()))
        conn.commit()
        cur.close()
        conn.close()


    def fp_compare(self, search_fp, match_fp):
        '''

        :param search_fp: 查询指纹
        :param match_fp: 库中指纹
        :return:最大相似值 float
        '''
        if len(search_fp) > len(match_fp):
            return 0
        max_similar = 0
        search_fp_len = len(search_fp)
        match_fp_len = len(match_fp)
        for i in range(match_fp_len - search_fp_len):
            temp = 0
            for j in range(search_fp_len):
                if match_fp[i + j] == search_fp[j]:
                    temp += 1
            if temp > max_similar:
                max_similar = temp
        return max_similar

    def search(self, path):
        '''
        搜索方法，输入为文件路径
        :param path: 待检索文件路径
        :return: 按照相似度排序后的列表，元素类型为tuple，二元组，歌曲名和相似匹配值
        '''
        #先计算出来我们的音频指纹
        v = music_regnization1.voice()
        v.loaddata(path)
        v.fft()
        #尝试连接数据库
        try:
            conn = MySQLdb.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                                   charset='utf8')
        except:
            raise IOError
            print('DataBase error2')
        cur = conn.cursor()
        cur.execute("SELECT * FROM test.music")
        result = cur.fetchall()
        compare_res = []
        for i in result:
            compare_res.append((self.fp_compare(v.high_point[:-1], eval(i[1])), i[0]))
        compare_res.sort(reverse=True)
        cur.close()
        conn.close()
        print (compare_res)
        return compare_res

    def search_and_play(self, path):
        '''
        搜索方法顺带了播放方法
        :param path:文件路径
        :return:
        '''
        v = music_regnization1.voice()
        v.loaddata(path)
        v.fft()
        try:
            conn = MySQLdb.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                                   charset='utf8')
        except:
            print ('DataBase error3')
            return None
        cur = conn.cursor()
        cur.execute("SELECT * FROM test.music")
        result = cur.fetchall()
        compare_res = []
        for i in result:
            compare_res.append((self.fp_compare(v.high_point[:-1], eval(i[1])), i[0]))
        compare_res.sort(reverse=True)
        cur.close()
        conn.close()
        print (compare_res)
        v.play(compare_res[0][1])
        return compare_res


if __name__ == '__main__':
    sss = memory('localhost', 3306, 'root', '123456', 'test')
    sss.addsong('1.mp3')
    sss.addsong('2.mp3')
    sss.addsong('3.mp3')
    sss.addsong('4.mp3')
    sss.addsong('5.mp3')


    sss.search_and_play('2.mp3')