from collections import namedtuple
import sys

input_pattern = namedtuple("input_pattern","name, input, desired_op")       

class BinaryData:
    def __init__(self, fileName):

        self.data = []
        count = 1 #for counting from 1-16 data patterns

        for line in open(fileName):
            line = line.strip()            
            line_contents = line.split(',')
            
            bits = []
            
            for i in range(0,5):
            	bits.append(int(line_contents[i]))

            ouput_desired = int(line_contents[5])
            #print bits
            self.data.append(input_pattern(count,bits,ouput_desired))
            count=count+1
           
