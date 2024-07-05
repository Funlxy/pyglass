
#include <iostream>
#include <string>
#include "glass/graph.hpp"
#include "glass/hnsw/hnsw.hpp"
#include "glass/nsg/nndescent.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/searcher.hpp"
#include <boost/asio.hpp> 
#include <boost/array.hpp> 
#include <iostream> 

enum { BUF_SIZE = 1024 };
using boost::asio::ip::tcp;


// 一次只能处理一个连接
class Session : public std::enable_shared_from_this<Session> {
public:
  Session(tcp::socket socket, std::shared_ptr<glass::SearcherBase> searcher) : socket_(std::move(socket)), searcher_(std::move(searcher)) {
  }

  void Start() {
    DoRead();
  }

  void DoRead() {
    // 得到对象指针
    auto self(shared_from_this()); // 保活,增加引用计数
    socket_.async_read_some(
        boost::asio::buffer(buffer_),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            DoWrite(length);
          }
        });
  }

  void DoWrite(std::size_t length) {
    auto self(shared_from_this());
    boost::asio::async_write(
        socket_,
        boost::asio::buffer(buffer_, length),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            DoRead(); // 又调用DoRead(),等待下一次读入
          }
        });
  }

private:
  tcp::socket socket_; // 当前连接
  std::array<char, BUF_SIZE> buffer_; // 缓存数据的buffer
  std::shared_ptr<glass::SearcherBase> searcher_; // index searcher

};

class Server {
public:
  Server(boost::asio::io_context& ioc, std::uint16_t port, std::unique_ptr<glass::SearcherBase> searcher)
      :acceptor_(ioc, tcp::endpoint(tcp::v4(), port)), searcher_(std::move(searcher)) {
        // searcher_->SetData(const float *data, int n, int dim)
  }
  void DoAccept() {
    acceptor_.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
          if (!ec) {
            std::make_shared<Session>(std::move(socket),searcher_)->Start(); // 创建一个session
            std::cout << "accept new socket\n";
          }
          DoAccept();
        });
  }
private:
  boost::asio::ip::tcp::acceptor acceptor_;
  std::shared_ptr<glass::SearcherBase> searcher_;
};

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <graph_file>" << std::endl;
    return 1;
  }
  glass::Graph<int> graph;
  graph.load(argv[2]);
  auto searcher = glass::create_searcher(graph, "L2", 2);
  // searcher.SetData()
  // searcher.Optimize();
  searcher->SetEf(32);
  std::uint16_t port = std::atoi(argv[1]);
  boost::asio::io_context ioc;
  Server server(ioc, port,std::move(searcher));
  server.DoAccept();
  ioc.run();
  return 0;
}