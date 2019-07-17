<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>ONNX pipeline</h1>
        <hr><br><br>
        <alert :message=message v-if="showMessage"></alert>

        <button type="button" class="btn btn-success btn-sm" v-b-modal.book-modal>Convert</button>
        <br><br>
        <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">Title</th>
              <th scope="col">Author</th>
              <th scope="col">Read?</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(book, index) in books" :key="index">
              <td>{{ book.title }}</td>
              <td>{{ book.author }}</td>
              <td>
                <span v-if="book.read">Yes</span>
                <span v-else>No</span>
              </td>
              <td>
                <button
                        type="button"
                        class="btn btn-warning btn-sm"
                        v-b-modal.book-update-modal
                        @click="editBook(book)">
                    Update
                </button>

                <button
                        type="button"
                        class="btn btn-danger btn-sm"
                        @click="onDeleteBook(book)">
                    Delete
                </button>

              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    <b-modal ref="addBookModal"
             id="book-modal"
             title="Add a new book"
             hide-footer>
      <b-form @submit="onSubmit" @reset="onReset" class="w-100">
      <b-form-group id="form-title-group"
                    label="Title:"
                    label-for="form-title-input">
          <b-form-input id="form-title-input"
                        type="text"
                        v-model="addBookForm.title"
                        required
                        placeholder="Enter title">
          </b-form-input>
        </b-form-group>
        <b-form-group id="form-author-group"
                      label="Author:"
                      label-for="form-author-input">
            <b-form-file id="form-author-input"
                          v-model="addBookForm.model"
                          required
                          placeholder="Choose a model..."
                          drop-placeholder="Drop model here...">
            </b-form-file>
          </b-form-group>
        <b-form-group id="form-read-group">
          <b-form-checkbox-group v-model="addBookForm.read" id="form-checks">
            <b-form-checkbox value="true">Read?</b-form-checkbox>
          </b-form-checkbox-group>
        </b-form-group>
        <b-button type="submit" variant="primary">Submit</b-button>
        <b-button type="reset" variant="danger">Reset</b-button>
      </b-form>
    </b-modal>
    <b-modal ref="editBookModal"
         id="book-update-modal"
         title="Update"
         hide-footer>
    <b-form @submit="onSubmitUpdate" @reset="onResetUpdate" class="w-100">
    <b-form-group id="form-title-edit-group"
                  label="Title:"
                  label-for="form-title-edit-input">
        <b-form-input id="form-title-edit-input"
                      type="text"
                      v-model="editForm.title"
                      required
                      placeholder="Enter title">
        </b-form-input>
      </b-form-group>
      <b-form-group id="form-author-edit-group"
                    label="Author:"
                    label-for="form-author-edit-input">
          <b-form-input id="form-author-edit-input"
                        type="text"
                        v-model="editForm.author"
                        required
                        placeholder="Enter author">
          </b-form-input>
        </b-form-group>
      <b-form-group id="form-read-edit-group">
        <b-form-checkbox-group v-model="editForm.read" id="form-checks">
          <b-form-checkbox value="true">Read?</b-form-checkbox>
        </b-form-checkbox-group>
      </b-form-group>
      <b-button type="submit" variant="primary">Update</b-button>
      <b-button type="reset" variant="danger">Cancel</b-button>
    </b-form>
  </b-modal>


  </div>
</template>

<script>
import axios from 'axios';
import Alert from './Alert';

export default {
  data() {
    return {
      books: [],
      addBookForm: {
        title: '',
        model: null,
        read: [],
      },
      editForm: {
        id: '',
        title: '',
        author: '',
        read: [],
      },
      message: '',
      showMessage: false,
    };
  },
  components: {
    alert: Alert,
  },
  methods: {
    getBooks() {
      const path = 'http://localhost:5000/books';
      axios.get(path)
        .then((res) => {
          this.books = res.data.books;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    convert(metadata, file) {
      const path = 'http://localhost:5000/convert';
      const json = JSON.stringify(metadata);

      const blob = new Blob([json], {
        type: 'application/json'
      });
      const data = new FormData();
      data.append("metadata", blob);
      data.append("file", file);

      axios.post(path, data)
      .then((res) => {
          console.log(res.data)
          if(res.data['status'] == 'success'){
            var data = res.data['logs'];
            this.message = res.data['logs'].replace("\n", "<br/>");
            this.showMessage = true;
          }
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getBooks();
        });
    },
    initForm() {
      this.addBookForm.title = '';
      this.addBookForm.file = null;
      this.addBookForm.read = [];
      this.editForm.id = '';
      this.editForm.title = '';
      this.editForm.author = '';
      this.editForm.read = [];
    },
    onSubmit(evt) {
      evt.preventDefault();
      this.$refs.addBookModal.hide();
      let read = false;
      if (this.addBookForm.read[0]) read = true;
      const metadata = {
        title: this.addBookForm.title,
        read, // property shorthand
      };
      this.convert(metadata, this.addBookForm.model);
      this.initForm();
    },
    onReset(evt) {
      evt.preventDefault();
      this.$refs.addBookModal.hide();
      this.initForm();
    },
    editBook(book) {
      this.editForm = book;
    },
    onSubmitUpdate(evt) {
      evt.preventDefault();
      this.$refs.editBookModal.hide();
      let read = false;
      if (this.editForm.read[0]) read = true;
      const payload = {
        title: this.editForm.title,
        author: this.editForm.author,
        read,
      };
      this.updateBook(payload, this.editForm.id);
    },
    updateBook(payload, bookID) {
      const path = `http://localhost:5000/books/${bookID}`;
      axios.put(path, payload)
        .then(() => {
          this.getBooks();
          this.message = 'Book updated!';
          this.showMessage = true;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
          this.getBooks();
        });
    },
    onResetUpdate(evt) {
      evt.preventDefault();
      this.$refs.editBookModal.hide();
      this.initForm();
      this.getBooks(); // why?
    },
    removeBook(bookID) {
      const path = `http://localhost:5000/books/${bookID}`;
      axios.delete(path)
        .then(() => {
          this.getBooks();
          this.message = 'Book removed!';
          this.showMessage = true;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
          this.getBooks();
        });
    },
    onDeleteBook(book) {
      this.removeBook(book.id);
    },

  },
  created() {
    this.getBooks();
  },
};
</script>
