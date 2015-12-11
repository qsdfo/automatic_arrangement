#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import xml.sax


class TotalLengthHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentElement = ""

        # Measure informations
        self.time = 0         # time counter. Tuple, each index is a voice
        self.beats = 0
        self.beats_set = False
        self.beat_type = 0
        self.beat_type_set = False
        self.total_length = 0
        self.total_length_list = []

    def startElement(self, tag, attributes):
        self.CurrentElement = tag

    def endElement(self, tag):
        if tag == 'measure':
            if self.beat_type_set and self.beats_set:
                self.time += self.beats * 4 / self.beat_type
            else:
                raise NameError('Beat-type or beats not set for the first measure of a part')

        if tag == 'part':
            self.total_length_list.append(self.time)
            # Re-initialize the different counters
            self.time = 0
            self.beats_set = False
            self.beat_type_set = False

        if tag == 'score-partwise':
            # Check if all the parts have the same length
            if len(set(self.total_length_list)) is not 1:
                raise NameError('All parts have not the same length')
            else:
                # Add a 1 at the end to allow the last note to stop
                self.total_length = self.total_length_list[1] + 1

    def characters(self, content):
        if content.strip():
            if self.CurrentElement == 'beats':
                self.beats = int(content)
                self.beats_set = True

            if self.CurrentElement == 'beat-type':
                self.beat_type = int(content)
                self.beat_type_set = True


if __name__ == "__main__":
    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler
    Handler = TotalLengthHandler()
    parser.setContentHandler(Handler)

    if(os.path.isfile("BeetAnGeSample.xml")):
        parser.parse("BeetAnGeSample.xml")
    else:
        print "Not a file"

    print Handler.total_length
