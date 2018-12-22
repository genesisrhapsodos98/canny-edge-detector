#pragma once
#include <string>

enum token_t
{
    Canny,
    Bad_command
};

class Lexer
{
  public:
    token_t lex(const char *s);
    Lexer();
    ~Lexer();
};