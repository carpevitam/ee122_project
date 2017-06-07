from math import sqrt
from scipy import misc
from cluster import Cluster
import random_color as rc
import numpy as np
# import matplotlib.pyplot as plt
import sys
import boto3
from time import sleep

DIFF_THRESH = 2
ITER_THRESH = 100


class MeanShift:

    def __init__(self, image_name, hs, hr):
        self.image = misc.imread(image_name)
        self.result = misc.imread(image_name)
        self.modes = dict()
        self.dim = self.image.shape[2] + 2  # this is the dimension of the feature space, typically 5 with (x, y, r, g, b)
        self.hr = hr  # color range
        self.hs = hs  # spatial range
        self.hr_filter = self.hr_filter()

    '''
    Calculate the difference between two feature space points.
    This function assumes v1 and v2 are of dtype int
    '''
    def diff(self, v1, v2):
        r = np.linalg.norm(v1-v2)
        return r

    '''
    return a function that filters out value greater than self.hr
    '''
    def hr_filter(self):
        def f(diff):
            if diff > self.hr:
                return 0
            else:
                return 1
        return np.vectorize(f)

    '''
    Calculate the mean around a vector in the feature space
    This function assumes vector dtype is int
    '''
    def mean(self, vector):

        # count = 0
        x = vector[0]
        y = vector[1]
        rgb = vector[2:]
        v_sum = np.zeros(self.dim).astype(int)

        # spatial window
        start_x = max(x - self.hs, 0)
        end_x = min(self.image.shape[0] - 1, x + self.hs)
        start_y = max(y - self.hs, 0)
        end_y = min(self.image.shape[1] - 1, y + self.hs)

        window_1 = self.image[start_x: end_x + 1, start_y: end_y + 1]
        window_2 = np.zeros((end_x - start_x + 1, end_y - start_y + 1, self.dim - 2))
        for i in range(window_2.shape[0]):
            for j in range(window_2.shape[1]):
                window_2[i, j] = rgb

        diff_window = np.apply_along_axis(np.linalg.norm, 2, window_1 - window_2)
        diff_window = self.hr_filter(diff_window)  # (i, j) = 1 if the point contribute to the mean

        count = 0

        for i in range(window_2.shape[0]):
            for j in range(window_2.shape[1]):
                if diff_window[i, j] == 1:
                    count += 1
                    v_sum += np.concatenate([[start_x + i, start_y + j], window_1[i, j]])

        if count == 0:
            return vector

        return np.divide(v_sum, count)

        # previous code
        # for i in range(start_x, end_x):
        #     for j in range(start_y, end_y):
        #         _rgb = self.image[i,j]
        #         diff = self.diff(_rgb, rgb)
        #         if diff <= self.hr:
        #             count += 1
        #             v_sum += np.concatenate([[i,j],_rgb])

        # if count == 0:
        #     return vector
        # else:
        #     return np.divide(v_sum, count)

    '''
    find the mode of the pixel (x, y) in the image
    using the mean shift algorithm
    '''
    def shift(self, x, y):
        iteration = 0
        cur = np.concatenate([[x, y], self.image[x, y]]).astype(int)
        next = self.mean(cur)
        while self.diff(cur, next) > DIFF_THRESH:
            cur = next
            next = self.mean(cur)
            iteration += 1
            if(iteration > ITER_THRESH):
                print 'ITER_THRESH reached when shifting ' + str(x) + ', ' + str(y)
                break
        return next

    '''
    Launch the process by first shifting every pixel to its mode,
    and then group similar modes into clusters.
    '''
    def run(self):
        print 'Begin shifting on ' + str(self.image.shape[0]) + ' * ' + str(self.image.shape[1]) + ' image. '
        print 'Paramter: hs = ' + str(self.hs) + '; hr = ' + str(self.hr) + '; dim = ' + str(self.dim)
        for i in range(self.image.shape[0]):
            print 'shifting row ' + str(i)
            for j in range(self.image.shape[1]):
                mode = self.shift(i, j)
                self.modes[(i, j)] = mode

        print 'Begin clustering.'

        clusters = []  # a list of Cluster object

        for i in range(self.image.shape[0]):
            print 'clustering row ' + str(i)
            for j in range(self.image.shape[1]):
                self.add_to_clusters(clusters, (i, j))

        print 'number of clusters: ' + str(len(clusters))

        colors = []

        for c in clusters:
            color = rc.generate_new_color(colors)
            colors.append(color)

            for i in range(3):
                color[i] = int(color[i] * 255)

            if self.dim == 4:
                color.append(255)

            for point in c.points:
                self.result[point[0], point[1]] = color

    # exact clustering - slow
    #
    # def belong_to_cluster(self, c, mode):
    #     x, y = mode[0], mode[1]
    #     rgb = mode[2:]
    #     for point in c:
    #         m = self.modes[point]
    #         m_x, m_y = m[0], m[1]
    #         m_rgb = m[2:]
    #         if self.diff(rgb, m_rgb) > self.hr:
    #             return False
    #     return True

    # def add_to_clusters(self, clusters, point):
    #     for c in clusters:
    #         if self.belong_to_cluster(c, self.modes[point]):
    #             c.append(point)
    #             return
    #     clusters.append([point])
    #     return

    # relaxed clustering

    def rgb_difference_to_cluster(self, c, mode):
        return self.diff(c.mean()[2:], mode[2:])

    def distance_to_cluster(self, c, mode):
        return self.diff(c.mean(), mode)

    def add_to_clusters(self, clusters, point):
        mode = self.modes[point]
        choice_1 = None  # closest cluster in terms of (x, y, r, g, b)
        choice_2 = None  # closest cluster in terms of (r, g, b)
        distance = sys.maxint
        rgb_distance = sys.maxint
        for c in clusters:
            d = self.distance_to_cluster(c, mode)
            rgb_d = self.rgb_difference_to_cluster(c, mode)
            if d < distance:
                distance = d
                choice_1 = c
            if rgb_d < rgb_distance:
                rgb_distance = rgb_d
                choice_2 = c

        # only consider rgb difference when determine whether to add to the cluster
        if choice_1 is not None and self.rgb_difference_to_cluster(choice_1, mode) < self.hr:
            choice_1.add(point, mode)
        elif choice_2 is not None and rgb_distance < self.hr:
            choice_2.add(point, mode)
        else:
            clusters.append(Cluster(point, mode))

    def save(self):
        misc.imsave('result.png', self.result)

    # def show(self):
    #     fig = plt.figure()
    #     a=fig.add_subplot(1,2,1)
    #     imgplot = plt.imshow(self.image)
    #     a.set_title('Original')
    #     a=fig.add_subplot(1,2,2)
    #     imgplot = plt.imshow(self.result)
    #     a.set_title('Segmented')
    #     plt.show()

# PARALLEL ----------------------------------------------------------------------|
    '''
	Parallelization of shifting code. Does shifting for rows [start, finish]
	and uploads them in a file referring to that machine number
    '''
    def shift_parallel(self, start, finish, machine_num):
        pass
        save_this = np.zeros((finish-start + 1, self.image.shape[1], self.dim))
        for i in range(start, finish + 1):
            print 'shifting row ' + str(i)
            send_message("shift_"+str(i))
            for j in range(self.image.shape[1]):
                mode = self.shift(i, j)
                save_this[i-start, j] = mode
        outfile = open("save_" + str(machine_num) + ".npy", 'w+')
        np.save(outfile, save_this)
        outfile.close()

    '''
	Same exact code as the original clustering code in run.
    '''
    def main_cluster(self):
        pass
        print 'Begin clustering.'
        clusters = []

        for i in range(self.image.shape[0]):
            print 'clustering row ' + str(i)
            if i % 10 == 0:
                send_message("cluster_"+str(i))
            for j in range(self.image.shape[1]):
                self.add_to_clusters(clusters, (i, j))

        print 'number of clusters: ' + str(len(clusters))

        for c in clusters:
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            for point in c:
                self.result[point[0], point[1]] = color
# END PARALLEL ------------------------------------------------------------------|

# UTILS -------------------------------------------------------------------------|
BUCKET = '122proj'
DIR_HOME = './'
def upload_file(filepath, name):
    s3 = boto3.client('s3')
    s3.upload_file(filepath, BUCKET, name)
def download_file(name, filepath):
    s3 = boto3.client('s3')
    s3.download_file(BUCKET, name, filepath)

Q = 'QURL'
def send_message(message):
    sqs = boto3.client('sqs')
    response = sqs.send_message(
        QueueUrl=Q,
        MessageBody=message
    )
# END UTILS ---------------------------------------------------------------------|

def main(argv):
    if (len(argv) != 2):
        print "usage: python mean_shift.py machine_num total_machines"
        return
    s3 = boto3.client('s3')
    machine_num = int(argv[0])
    total_machines = int(argv[1])
    total_set = set()

    def idle(image_set):
        while True:
            print 'z',
            if 'Contents' in s3.list_objects(Bucket=BUCKET):
                files_list = s3.list_objects(Bucket=BUCKET)['Contents']
                for file in files_list:
                    if file['Key'][-4:] == '.jpg' and file['Key'] not in image_set:
                        image_set.add(file['Key'])
                        image_name = file['Key']
                        print image_name
                        download_file(file['Key'], DIR_HOME + file['Key'])
                        return image_name
            sleep(3)

    def run(image_name):
        print image_name
        m = MeanShift(image_name, hs=10, hr=30)

        chunk_size = m.image.shape[0] / float(total_machines)
        start = int(machine_num * chunk_size)
        finish = int((machine_num + 1) * chunk_size) - 1
        m.shift_parallel(start, finish, machine_num)
        upload_file("save_" + str(machine_num) + ".npy", "save_" + str(machine_num) + ".npy")

        if machine_num != 0:
            return

        downloaded_nums = set()
        while len(downloaded_nums) < total_machines:
            print len(downloaded_nums)
            print total_machines
            print "X"
            if 'Contents' in s3.list_objects(Bucket=BUCKET):
                files_list = s3.list_objects(Bucket=BUCKET)['Contents']
                for file in files_list:
                    if file['Key'][:5] == 'save_' and file['Key'] not in downloaded_nums:
                        downloaded_nums.add(file['Key'])
                        download_file(file['Key'], DIR_HOME + file['Key'])
                        print "donwloaded " + file['Key']
            print len(downloaded_nums)
            print total_machines
            print "Y"
            sleep(3)

        for i in range(total_machines):
            x = np.load('save_' + str(i) + '.npy')
            spec_start = int(i * chunk_size)
            spec_finish = int((i + 1) * chunk_size) - 1
            for i in range(spec_finish - spec_start+1):
                print "i= " + str(i) + " ",
                for j in range(m.image.shape[1]):
                    m.modes[(i+spec_start, j)] = x[i, j]
        m.main_cluster()
        m.save()
        upload_file(image_name[:-4] + ".png", image_name[:-4] + ".png")

    while True:
        job_image = idle(total_set)
        run(job_image)

if __name__ == "__main__":
    main(sys.argv[1:])