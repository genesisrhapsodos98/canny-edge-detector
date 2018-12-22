#include "Lexer.h"

Lexer::Lexer()
{
}

Lexer::~Lexer()
{
}

token_t Lexer::lex(const char *s)
{
    static struct entryStrings
    {
        const char *keys;
        token_t token;
    } token_table[] = {
        {"--canny", Canny},
        {NULL, Bad_command}};

    struct entryStrings *p = token_table;
    for (; p->keys != NULL && strcmp(p->keys, s) != 0; ++p)
        ;
    return p->token;
}