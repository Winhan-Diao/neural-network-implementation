#pragma once
#include <iostream>
#include <cstring>

void inputTill(std::istream& is,  std::string& buffer, const std::string& delimiter) {
    static constexpr ssize_t tmp_buffer_size = 2048;
    ssize_t delimiter_matched_pos = 0;
    ssize_t delimiter_end_pos = delimiter.length();
    char tmp_buffer[tmp_buffer_size + 1]{'\0'};
    bool finish = false;
    do {
        for (ssize_t i = 0;  i < tmp_buffer_size && (is.peek() != std::istream::traits_type::eof()); ++i) {
            is.read(&tmp_buffer[i], 1);
            if (delimiter[delimiter_matched_pos] == tmp_buffer[i]) {
                ++delimiter_matched_pos;
                if (delimiter_matched_pos == delimiter_end_pos) {
                    finish = true;
                    break;
                }
            } else {
                delimiter_matched_pos = 0;
            }
        }
        buffer += tmp_buffer;
        std::memset(tmp_buffer, '\0', tmp_buffer_size);
    } while (!finish);
} 

void skipTill(std::istream& is, const std::string& delimiter) {
    ssize_t delimiter_matched_pos = 0;
    ssize_t delimiter_end_pos = delimiter.length();
    char tmp;
    do {
        if ((tmp = is.peek()) != std::istream::traits_type::eof()) {
            is.ignore();
            if (delimiter[delimiter_matched_pos] == tmp) {
                ++delimiter_matched_pos;
                if (delimiter_matched_pos == delimiter_end_pos) {
                    break;
                }
            } else {
                delimiter_matched_pos = 0;
            }
        } else {
            break;
        }

    } while (1);
}
